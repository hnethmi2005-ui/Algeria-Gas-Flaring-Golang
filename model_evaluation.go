
package main

import (
    "encoding/csv"
    "fmt"
    "image/color"
    "log"
    "math"
    "os"
    "sort"
    "strconv"

    "gonum.org/v1/gonum/stat"
    "gonum.org/v1/plot"
    "gonum.org/v1/plot/plotter"
    "gonum.org/v1/plot/vg"
)

// Store model evaluation results
type Metrics struct {
    Name string
    MAE  float64
    MSE  float64
    RMSE float64
    R2   float64
}

// Calculate regression evaluation metrics
func evaluate(name string, actual, predicted []float64) Metrics {

    n := float64(len(actual))
    mean := stat.Mean(actual, nil)

    absoluteError := 0.0
    squaredError := 0.0
    totalVariation := 0.0

    for i := range actual {

        errorValue := actual[i] - predicted[i]

        absoluteError += math.Abs(errorValue)
        squaredError += errorValue * errorValue

        difference := actual[i] - mean
        totalVariation += difference * difference
    }

    mse := squaredError / n

    r2 := math.NaN()

    if totalVariation > 0 {
        r2 = 1 - squaredError/totalVariation
    }

    return Metrics{
        Name: name,
        MAE: absoluteError / n,
        MSE: mse,
        RMSE: math.Sqrt(mse),
        R2: r2,
    }
}

// Save figures
func save(p *plot.Plot, filename string) {

    err := p.Save(
        8*vg.Inch,
        5*vg.Inch,
        "figures/"+filename+".png",
    )

    if err != nil {
        log.Fatal(err)
    }

    fmt.Println("Saved:", filename+".png")
}

// Model comparison bar chart
func comparisonChart(
    names []string,
    scores []float64,
    title, ylabel, filename string,
) {

    p := plot.New()
    p.Title.Text = title
    p.Y.Label.Text = ylabel

    colours := []color.Color{
        color.RGBA{70, 130, 180, 255},
        color.RGBA{255, 140, 0, 255},
        color.RGBA{46, 160, 90, 255},
    }

    for i, score := range scores {

        if math.IsNaN(score) {
            log.Fatal("Undefined metric:", title)
        }

        bars, err := plotter.NewBarChart(
            plotter.Values{score},
            vg.Points(45),
        )

        if err != nil {
            log.Fatal(err)
        }

        bars.XMin = float64(i)
        bars.Color = colours[i]

        p.Add(bars)
    }

    p.NominalX(names...)
    p.Add(plotter.NewGrid())

    save(p, filename)
}

// Actual versus predicted scatter plot
func predictionPlot(
    actual, predicted []float64,
    name, filename string,
) {

    p := plot.New()

    p.Title.Text = name + ": Actual vs Predicted"
    p.X.Label.Text = "Actual Volume (million m3)"
    p.Y.Label.Text = "Predicted Volume (million m3)"

    points := make(plotter.XYs, len(actual))

    minValue := math.Min(actual[0], predicted[0])
    maxValue := math.Max(actual[0], predicted[0])

    for i := range actual {

        points[i].X = actual[i]
        points[i].Y = predicted[i]

        minValue = math.Min(
            minValue,
            math.Min(actual[i], predicted[i]),
        )

        maxValue = math.Max(
            maxValue,
            math.Max(actual[i], predicted[i]),
        )
    }

    scatter, err := plotter.NewScatter(points)

    if err != nil {
        log.Fatal(err)
    }

    scatter.GlyphStyle.Color = color.RGBA{
        70, 130, 180, 255,
    }

    scatter.GlyphStyle.Radius = vg.Points(2.5)

    // Perfect prediction reference line
    reference := plotter.XYs{
        {X: minValue, Y: minValue},
        {X: maxValue, Y: maxValue},
    }

    line, err := plotter.NewLine(reference)

    if err != nil {
        log.Fatal(err)
    }

    line.Color = color.RGBA{220, 50, 50, 255}
    line.Width = vg.Points(2)

    p.Add(scatter, line)

    p.X.Min, p.X.Max = minValue, maxValue
    p.Y.Min, p.Y.Max = minValue, maxValue

    p.Legend.Add("Predictions", scatter)
    p.Legend.Add("Perfect Prediction", line)

    save(p, filename)
}

// Residual scatter plot
func residualPlot(
    actual, predicted []float64,
    name, filename string,
) []float64 {

    p := plot.New()

    p.Title.Text = name + ": Residual Plot"
    p.X.Label.Text = "Predicted Volume (million m3)"
    p.Y.Label.Text = "Residual (million m3)"

    residuals := make([]float64, len(actual))
    points := make(plotter.XYs, len(actual))

    minX := predicted[0]
    maxX := predicted[0]

    for i := range actual {

        residuals[i] = actual[i] - predicted[i]

        points[i].X = predicted[i]
        points[i].Y = residuals[i]

        minX = math.Min(minX, predicted[i])
        maxX = math.Max(maxX, predicted[i])
    }

    scatter, err := plotter.NewScatter(points)

    if err != nil {
        log.Fatal(err)
    }

    scatter.GlyphStyle.Color = color.RGBA{
        138, 43, 226, 255,
    }

    scatter.GlyphStyle.Radius = vg.Points(2.5)

    // Zero residual reference line
    reference := plotter.XYs{
        {X: minX, Y: 0},
        {X: maxX, Y: 0},
    }

    line, err := plotter.NewLine(reference)

    if err != nil {
        log.Fatal(err)
    }

    line.Color = color.RGBA{220, 50, 50, 255}
    line.Width = vg.Points(2)

    p.Add(scatter, line)

    save(p, filename)

    return residuals
}

// Residual distribution histogram
func errorHistogram(
    residuals []float64,
    name, filename string,
) {

    p := plot.New()

    p.Title.Text = name + ": Residual Distribution"
    p.X.Label.Text = "Residual (million m3)"
    p.Y.Label.Text = "Frequency"

    histogram, err := plotter.NewHist(
        plotter.Values(residuals), 25,
    )

    if err != nil {
        log.Fatal(err)
    }

    histogram.FillColor = color.RGBA{
        46, 160, 130, 255,
    }

    p.Add(histogram)

    save(p, filename)
}

// Save model evaluation results
func saveResults(results []Metrics) {

    file, err := os.Create(
        "outputs/model_evaluation.csv",
    )

    if err != nil {
        log.Fatal(err)
    }

    defer file.Close()

    writer := csv.NewWriter(file)

    if err := writer.Write([]string{
        "Model", "MAE", "MSE", "RMSE", "R2",
    }); err != nil {
        log.Fatal(err)
    }

    for _, m := range results {

        row := []string{
            m.Name,
            strconv.FormatFloat(m.MAE, 'f', 6, 64),
            strconv.FormatFloat(m.MSE, 'f', 6, 64),
            strconv.FormatFloat(m.RMSE, 'f', 6, 64),
            strconv.FormatFloat(m.R2, 'f', 6, 64),
        }

        if err := writer.Write(row); err != nil {
            log.Fatal(err)
        }
    }

    writer.Flush()

    if err := writer.Error(); err != nil {
        log.Fatal(err)
    }

    fmt.Println("Saved: outputs/model_evaluation.csv")
}

func main() {

    // Create output directories
    for _, dir := range []string{"figures", "outputs"} {
        if err := os.MkdirAll(dir, 0755); err != nil {
            log.Fatal(err)
        }
    }

    // Load saved testing predictions
    file, err := os.Open(
        "outputs/test_predictions.csv",
    )

    if err != nil {
        log.Fatal(err)
    }

    defer file.Close()

    records, err := csv.NewReader(file).ReadAll()

    if err != nil {
        log.Fatal(err)
    }

    if len(records) < 2 {
        log.Fatal("Prediction dataset is empty.")
    }

    expected := []string{
        "Actual",
        "Linear_Regression",
        "Decision_Tree",
        "Random_Forest",
    }

    if len(records[0]) != len(expected) {
        log.Fatal("Unexpected number of columns.")
    }

    for i, name := range expected {
        if records[0][i] != name {
            log.Fatal("Unexpected column:", records[0][i])
        }
    }

    actual := []float64{}
    predictions := make([][]float64, 3)

    for _, row := range records[1:] {

        values := make([]float64, 4)

        for j := 0; j < 4; j++ {

            value, err := strconv.ParseFloat(row[j], 64)

            if err != nil ||
                math.IsNaN(value) ||
                math.IsInf(value, 0) {

                log.Fatal("Invalid prediction value.")
            }

            values[j] = value
        }

        actual = append(actual, values[0])

        for j := 0; j < 3; j++ {
            predictions[j] = append(
                predictions[j], values[j+1],
            )
        }
    }

    names := []string{
        "Linear Regression",
        "Decision Tree",
        "Random Forest",
    }

    shortNames := []string{"MLR", "DTR", "RFR"}

    filenames := []string{
        "linear_regression",
        "decision_tree",
        "random_forest",
    }

    results := []Metrics{}

    fmt.Println("========== TEST DATA SUMMARY ==========")
    fmt.Println("Testing Observations:", len(actual))

    // Evaluate all three models
    for i, name := range names {

        metrics := evaluate(
            name, actual, predictions[i],
        )

        results = append(results, metrics)
    }

    // Display evaluation table
    fmt.Println("\n========== MODEL PERFORMANCE ==========")

    fmt.Printf(
        "%-22s %12s %12s %12s %12s\n",
        "Model", "MAE", "MSE", "RMSE", "R2",
    )

    for _, m := range results {

        fmt.Printf(
            "%-22s %12.4f %12.4f %12.4f %12.4f\n",
            m.Name, m.MAE, m.MSE, m.RMSE, m.R2,
        )
    }

    // Collect metric values
    mae := []float64{}
    mse := []float64{}
    rmse := []float64{}
    r2 := []float64{}

    for _, m := range results {

        mae = append(mae, m.MAE)
        mse = append(mse, m.MSE)
        rmse = append(rmse, m.RMSE)
        r2 = append(r2, m.R2)
    }

    // Generate comparison charts
    fmt.Println("\n========== PERFORMANCE CHARTS ==========")

    comparisonChart(
        shortNames, mae,
        "Mean Absolute Error Comparison",
        "MAE (million m3)", "mae_comparison",
    )

    comparisonChart(
        shortNames, mse,
        "Mean Squared Error Comparison",
        "MSE (million m3)^2", "mse_comparison",
    )

    comparisonChart(
        shortNames, rmse,
        "Root Mean Squared Error Comparison",
        "RMSE (million m3)", "rmse_comparison",
    )

    comparisonChart(
        shortNames, r2,
        "R-Squared Comparison",
        "R2 Score", "r2_comparison",
    )

    // Prediction and residual analysis
    fmt.Println("\n========== RESIDUAL ANALYSIS ==========")

    for i, name := range names {

        predictionPlot(
            actual,
            predictions[i],
            name,
            filenames[i]+"_actual_predicted",
        )

        residuals := residualPlot(
            actual,
            predictions[i],
            name,
            filenames[i]+"_residuals",
        )

        errorHistogram(
            residuals,
            name,
            filenames[i]+"_error_distribution",
        )

        fmt.Printf(
            "%s | Mean Residual: %.4f | Residual SD: %.4f\n",
            name,
            stat.Mean(residuals, nil),
            stat.StdDev(residuals, nil),
        )
    }

    // Rank models by RMSE
    sort.SliceStable(results, func(i, j int) bool {
        return results[i].RMSE < results[j].RMSE
    })

    fmt.Println("\n========== MODEL RANKING ==========")

    for i, m := range results {

        fmt.Printf(
            "%d. %s | RMSE: %.4f | R2: %.4f\n",
            i+1, m.Name, m.RMSE, m.R2,
        )
    }

    best := results[0]

    fmt.Println("\n========== BEST-PERFORMING MODEL ==========")

    fmt.Println("Selected Model:", best.Name)
    fmt.Printf("MAE: %.4f\n", best.MAE)
    fmt.Printf("MSE: %.4f\n", best.MSE)
    fmt.Printf("RMSE: %.4f\n", best.RMSE)
    fmt.Printf("R2: %.4f\n", best.R2)

    // Save evaluation results
    saveResults(results)

    summary := fmt.Sprintf(
        "Best Model by RMSE: %s\nMAE: %.6f\nMSE: %.6f\nRMSE: %.6f\nR2: %.6f\n",
        best.Name, best.MAE, best.MSE,
        best.RMSE, best.R2,
    )

    err = os.WriteFile(
        "outputs/best_model_summary.txt",
        []byte(summary), 0644,
    )

    if err != nil {
        log.Fatal(err)
    }

    fmt.Println("\nModel evaluation completed successfully.")
    fmt.Println("All 13 evaluation figures saved.")
}
