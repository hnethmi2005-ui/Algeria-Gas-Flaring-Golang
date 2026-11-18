
package main

import (
    "encoding/csv"
    "encoding/json"
    "fmt"
    "image/color"
    "log"
    "math"
    "math/rand"
    "os"
    "sort"
    "strconv"
    "strings"

    "gonum.org/v1/gonum/stat"
    "gonum.org/v1/plot"
    "gonum.org/v1/plot/plotter"
    "gonum.org/v1/plot/vg"
)

// Saved Random Forest structure
type TreeNode struct {
    Feature   int
    Threshold float64
    Value     float64
    Left      *TreeNode
    Right     *TreeNode
}

type ForestModel struct {
    Trees []*TreeNode
}

// Preprocessing parameters
type Preprocessor struct {
    NumericalNames   []string
    CategoricalNames []string
    Means            []float64
    StdDevs          []float64
    Categories       [][]string
}

// Dataset structure
type Dataset struct {
    X        [][]float64
    Y        []float64
    Features []string
}

// Group related encoded features
type FeatureGroup struct {
    Name    string
    Columns []int
}

type Importance struct {
    Name  string
    Score float64
    SD    float64
}

// Load JSON files
func loadJSON(path string, target interface{}) {

    content, err := os.ReadFile(path)

    if err != nil {
        log.Fatal(err)
    }

    if err := json.Unmarshal(content, target); err != nil {
        log.Fatal(err)
    }
}

// Load processed datasets
func loadData(path string) Dataset {

    file, err := os.Open(path)

    if err != nil {
        log.Fatal(err)
    }

    defer file.Close()

    records, err := csv.NewReader(file).ReadAll()

    if err != nil {
        log.Fatal(err)
    }

    if len(records) < 2 {
        log.Fatal("Empty dataset:", path)
    }

    last := len(records[0])-1

    if records[0][last] != "Flaring Vol (million m3)" {
        log.Fatal("Unexpected target column")
    }

    data := Dataset{
        Features: records[0][:last],
    }

    for _, row := range records[1:] {

        values := make([]float64, len(row))

        for j, text := range row {

            value, err := strconv.ParseFloat(text, 64)

            if err != nil || math.IsNaN(value) ||
                math.IsInf(value, 0) {
                log.Fatal("Invalid value:", text)
            }

            values[j] = value
        }

        data.X = append(data.X, values[:last])
        data.Y = append(data.Y, values[last])
    }

    return data
}

// Decision Tree prediction
func predictTree(node *TreeNode, x []float64) float64 {

    if node.Feature == -1 {
        return node.Value
    }

    if x[node.Feature] <= node.Threshold {
        return predictTree(node.Left, x)
    }

    return predictTree(node.Right, x)
}

// Random Forest prediction
func predictForest(model ForestModel, x []float64) float64 {

    total := 0.0

    for _, tree := range model.Trees {
        total += predictTree(tree, x)
    }

    return total / float64(len(model.Trees))
}

// Calculate RMSE
func calculateRMSE(actual, predicted []float64) float64 {

    total := 0.0

    for i := range actual {

        difference := actual[i] - predicted[i]
        total += difference * difference
    }

    return math.Sqrt(total / float64(len(actual)))
}

// Generate predictions
func predictAll(model ForestModel, X [][]float64) []float64 {

    predictions := make([]float64, len(X))

    for i, row := range X {
        predictions[i] = predictForest(model, row)
    }

    return predictions
}

// Group original predictors and one-hot columns
func createGroups(features []string) []FeatureGroup {

    groups := []FeatureGroup{}
    index := make(map[string]int)

    for j, feature := range features {

        name := strings.SplitN(feature, "=", 2)[0]

        if position, exists := index[name]; exists {

            groups[position].Columns = append(
                groups[position].Columns, j,
            )

        } else {

            index[name] = len(groups)

            groups = append(groups, FeatureGroup{
                Name: name,
                Columns: []int{j},
            })
        }
    }

    return groups
}

// Save CSV analytical results
func saveCSV(path string, rows [][]string) {

    file, err := os.Create(path)

    if err != nil {
        log.Fatal(err)
    }

    defer file.Close()

    writer := csv.NewWriter(file)

    if err := writer.WriteAll(rows); err != nil {
        log.Fatal(err)
    }

    fmt.Println("Saved:", path)
}

// Format numerical results
func number(value float64) string {
    return strconv.FormatFloat(value, 'f', 8, 64)
}

// Save generated figures
func savePlot(p *plot.Plot, name string) {

    path := "figures/" + name + ".png"

    if err := p.Save(8*vg.Inch, 5*vg.Inch, path); err != nil {
        log.Fatal(err)
    }

    fmt.Println("Saved:", path)
}

// Generate explanation bar charts
func barChart(
    names []string,
    values []float64,
    title, ylabel, filename string,
) {

    p := plot.New()

    p.Title.Text = title
    p.Y.Label.Text = ylabel

    for i, value := range values {

        bar, err := plotter.NewBarChart(
            plotter.Values{value}, vg.Points(40),
        )

        if err != nil {
            log.Fatal(err)
        }

        bar.XMin = float64(i)

        if value >= 0 {
            bar.Color = color.RGBA{210, 75, 110, 255}
        } else {
            bar.Color = color.RGBA{60, 120, 210, 255}
        }

        p.Add(bar)
    }

    p.NominalX(names...)
    p.Add(plotter.NewGrid())

    savePlot(p, filename)
}

// 1. Permutation Feature Importance
func permutationImportance(
    model ForestModel,
    test Dataset,
    groups []FeatureGroup,
    baseline float64,
) {

    rng := rand.New(rand.NewSource(42))

    results := []Importance{}
    repeats := 10

    for _, group := range groups {

        changes := []float64{}

        for r := 0; r < repeats; r++ {

            permutation := rng.Perm(len(test.X))
            shuffled := make([][]float64, len(test.X))

            for i, row := range test.X {

                shuffled[i] = append([]float64(nil), row...)

                // Shuffle grouped columns together
                for _, j := range group.Columns {
                    shuffled[i][j] = test.X[permutation[i]][j]
                }
            }

            predictions := predictAll(model, shuffled)

            newRMSE := calculateRMSE(test.Y, predictions)

            changes = append(changes, newRMSE-baseline)
        }

        results = append(results, Importance{
            Name: group.Name,
            Score: stat.Mean(changes, nil),
            SD: stat.StdDev(changes, nil),
        })
    }

    // Rank features by importance
    sort.SliceStable(results, func(i, j int) bool {
        return results[i].Score > results[j].Score
    })

    rows := [][]string{
        {"Feature", "Mean_RMSE_Increase", "StdDev"},
    }

    names := []string{}
    scores := []float64{}

    fmt.Println("\n========== PERMUTATION IMPORTANCE ==========")

    for i, result := range results {

        fmt.Printf("%d. %-15s Importance: %.4f | SD: %.4f\n",
            i+1, result.Name, result.Score, result.SD)

        names = append(names, result.Name)
        scores = append(scores, result.Score)

        rows = append(rows, []string{
            result.Name,
            number(result.Score),
            number(result.SD),
        })
    }

    saveCSV("outputs/permutation_importance.csv", rows)

    barChart(
        names, scores,
        "Permutation Feature Importance",
        "Mean Increase in RMSE",
        "xai_feature_importance",
    )

    fmt.Println("Highest-ranked predictor:", results[0].Name)
}

// 2. Partial Dependence Plots
func partialDependence(
    model ForestModel,
    train Dataset,
    test Dataset,
    prep Preprocessor,
) {

    fmt.Println("\n========== PARTIAL DEPENDENCE ==========")

    for j, feature := range prep.NumericalNames {

        column := -1

        for k, name := range train.Features {
            if name == feature {
                column = k
                break
            }
        }

        if column == -1 {
            log.Fatal("Missing PDP feature:", feature)
        }

        values := []float64{}

        for _, row := range train.X {
            values = append(values, row[column])
        }

        sort.Float64s(values)

        // Use the middle 90% of training feature values
        lower := stat.Quantile(
            0.05, stat.Empirical, values, nil,
        )

        upper := stat.Quantile(
            0.95, stat.Empirical, values, nil,
        )

        grid := 25

        if lower == upper {
            grid = 1
        }

        points := make(plotter.XYs, grid)

        rows := [][]string{
            {"Feature_Value", "Average_Prediction"},
        }

        for k := 0; k < grid; k++ {

            scaled := lower

            if grid > 1 {
                scaled = lower +
                    (upper-lower)*float64(k)/float64(grid-1)
            }

            total := 0.0

            for _, row := range test.X {

                modified := append([]float64(nil), row...)
                modified[column] = scaled

                total += predictForest(model, modified)
            }

            average := total / float64(len(test.X))

            // Convert feature back to its original scale
            original := scaled*prep.StdDevs[j] + prep.Means[j]

            points[k].X = original
            points[k].Y = average

            rows = append(rows, []string{
                number(original),
                number(average),
            })
        }

        p := plot.New()

        p.Title.Text = "Partial Dependence: " + feature
        p.X.Label.Text = feature
        p.Y.Label.Text = "Average Prediction (million m3)"

        line, err := plotter.NewLine(points)

        if err != nil {
            log.Fatal(err)
        }

        line.Color = color.RGBA{40, 150, 90, 255}
        line.Width = vg.Points(2)

        scatter, err := plotter.NewScatter(points)

        if err != nil {
            log.Fatal(err)
        }

        scatter.GlyphStyle.Color = line.Color
        scatter.GlyphStyle.Radius = vg.Points(2)

        p.Add(line, scatter, plotter.NewGrid())

        filename := "xai_pdp_" + strings.ToLower(feature)

        savePlot(p, filename)
        saveCSV("outputs/"+filename+".csv", rows)

        fmt.Printf("%s: PDP generated using training range\n",
            feature)
    }
}

// 3. Model-agnostic approximate Shapley values
func approximateShapley(
    model ForestModel,
    train Dataset,
    test Dataset,
    groups []FeatureGroup,
) {

    fmt.Println("\n========== APPROXIMATE SHAPLEY VALUES ==========")

    rng := rand.New(rand.NewSource(42))

    // Random training background samples
    backgroundCount := 50
    background := make([][]float64, backgroundCount)

    for i := range background {

        background[i] = train.X[rng.Intn(len(train.X))]
    }

    baseline := stat.Mean(
        predictAll(model, background), nil,
    )

    // Explain three representative prediction levels
    predictions := predictAll(model, test.X)

    indices := make([]int, len(test.X))

    for i := range indices {
        indices[i] = i
    }

    sort.SliceStable(indices, func(i, j int) bool {
        return predictions[indices[i]] <
               predictions[indices[j]]
    })

    selected := []int{
        indices[len(indices)/10],
        indices[len(indices)/2],
        indices[9*len(indices)/10],
    }

    labels := []string{"low", "median", "high"}

    rows := [][]string{
        {
            "Record", "Prediction_Level", "Feature",
            "Contribution", "Baseline",
            "Prediction", "Actual",
        },
    }

    trials := 200

    for s, record := range selected {

        x := test.X[record]

        contributions := make([]float64, len(groups))

        sampledBaseline := 0.0

        // Estimate marginal feature contributions
        for trial := 0; trial < trials; trial++ {

            reference := background[trial%backgroundCount]

            current := append([]float64(nil), reference...)

            previous := predictForest(model, current)

            sampledBaseline += previous

            order := rng.Perm(len(groups))

            for _, g := range order {

                // Replace all columns of one feature group
                for _, column := range groups[g].Columns {
                    current[column] = x[column]
                }

                updated := predictForest(model, current)

                contributions[g] += updated-previous

                previous = updated
            }
        }

        sampledBaseline /= float64(trials)

        names := []string{}
        total := 0.0

        fmt.Printf("\n--- %s prediction: Record %d ---\n",
            labels[s], record+1)

        fmt.Printf("Actual: %.4f\n", test.Y[record])
        fmt.Printf("Prediction: %.4f\n", predictions[record])
        fmt.Printf("Baseline: %.4f\n", sampledBaseline)

        for g, group := range groups {

            contributions[g] /= float64(trials)

            total += contributions[g]

            names = append(names, group.Name)

            fmt.Printf("%-15s Contribution: %+.4f\n",
                group.Name, contributions[g])

            rows = append(rows, []string{
                strconv.Itoa(record+1),
                labels[s],
                group.Name,
                number(contributions[g]),
                number(sampledBaseline),
                number(predictions[record]),
                number(test.Y[record]),
            })
        }

        reconstructed := sampledBaseline + total

        fmt.Printf("Reconstructed Prediction: %.4f\n",
            reconstructed)

        fmt.Printf("Additivity Difference: %.10f\n",
            predictions[record]-reconstructed)

        barChart(
            names,
            contributions,
            "Approximate Shapley Values: "+labels[s],
            "Feature Contribution (million m3)",
            "xai_shapley_"+labels[s],
        )
    }

    fmt.Printf("\nBackground Mean Prediction: %.4f\n",
        baseline)

    fmt.Println("Background Samples:", backgroundCount)
    fmt.Println("Permutations per Explanation:", trials)

    saveCSV("outputs/approximate_shapley_values.csv", rows)
}

func main() {

    // Create required output folders
    for _, dir := range []string{"figures", "outputs"} {
        if err := os.MkdirAll(dir, 0755); err != nil {
            log.Fatal(err)
        }
    }

    // Load trained model and preprocessing information
    var model ForestModel
    var prep Preprocessor
    var features []string

    loadJSON("models/random_forest.json", &model)

    loadJSON(
        "models/preprocessing_parameters.json", &prep,
    )

    loadJSON("models/feature_names.json", &features)

    if len(model.Trees) == 0 {
        log.Fatal("Random Forest contains no trees")
    }

    train := loadData("datasets/train_data.csv")
    test := loadData("datasets/test_data.csv")

    // Verify feature order
    expected := strings.Join(features, "\x00")

    if strings.Join(train.Features, "\x00") != expected ||
       strings.Join(test.Features, "\x00") != expected {

        log.Fatal("Feature names or order do not match")
    }

    groups := createGroups(features)

    // Verify baseline prediction performance
    predictions := predictAll(model, test.X)

    baseline := calculateRMSE(test.Y, predictions)

    fmt.Println("========== XAI MODEL SUMMARY ==========")

    fmt.Println("Random Forest Trees:", len(model.Trees))
    fmt.Println("Training Observations:", len(train.X))
    fmt.Println("Testing Observations:", len(test.X))
    fmt.Println("Encoded Features:", len(features))
    fmt.Println("Original Feature Groups:", len(groups))

    fmt.Printf("Baseline RMSE: %.4f\n", baseline)

    // Run all XAI techniques
    permutationImportance(model, test, groups, baseline)

    partialDependence(model, train, test, prep)

    approximateShapley(model, train, test, groups)

    fmt.Println("\n========== XAI SUMMARY ==========")
    fmt.Println("Permutation Feature Importance: Completed")
    fmt.Println("Partial Dependence Analysis: Completed")
    fmt.Println("Approximate Shapley Analysis: Completed")
    fmt.Println("Total XAI Figures Generated: 7")
    fmt.Println("Random Seed: 42")

    fmt.Println("\nExplainable AI analysis completed successfully.")
}
