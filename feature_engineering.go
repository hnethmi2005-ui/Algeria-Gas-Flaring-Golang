
package main

import (
    "encoding/csv"
    "encoding/json"
    "fmt"
    "log"
    "math"
    "math/rand"
    "os"
    "sort"
    "strconv"
    "strings"

    "gonum.org/v1/gonum/stat"
)

// Structure for dataset observations
type Observation struct {
    Numerical   []float64
    Categorical []string
    Target      float64
    Group       string
}

// Store preprocessing parameters
type Preprocessor struct {
    NumericalNames   []string
    CategoricalNames []string
    Means            []float64
    StdDevs          []float64
    Categories       [][]string
}

// Save processed dataset
func saveDataset(
    filename string,
    data []Observation,
    prep Preprocessor,
) {
    file, err := os.Create(filename)
    if err != nil {
        log.Fatal(err)
    }
    defer file.Close()

    writer := csv.NewWriter(file)

    // Create feature names
    headers := append([]string{}, prep.NumericalNames...)

    for j, name := range prep.CategoricalNames {
        for _, category := range prep.Categories[j] {
            headers = append(headers, name+"="+category)
        }
        headers = append(headers, name+"=__UNSEEN__")
    }

    headers = append(headers, "Flaring Vol (million m3)")

    if err := writer.Write(headers); err != nil {
        log.Fatal(err)
    }

    // Transform each observation
    for _, obs := range data {

        row := []string{}

        // Standardise numerical features
        for j, value := range obs.Numerical {
            scaled := (value - prep.Means[j]) /
                prep.StdDevs[j]

            row = append(row,
                strconv.FormatFloat(scaled, 'f', 8, 64))
        }

        // Apply one-hot encoding
        for j, value := range obs.Categorical {
            matched := false

            for _, category := range prep.Categories[j] {
                if value == category {
                    row = append(row, "1")
                    matched = true
                } else {
                    row = append(row, "0")
                }
            }

            // Handle categories unseen during training
            if matched {
                row = append(row, "0")
            } else {
                row = append(row, "1")
            }
        }

        // Keep target in its original measurement unit
        row = append(row,
            strconv.FormatFloat(obs.Target, 'g', -1, 64))

        if err := writer.Write(row); err != nil {
            log.Fatal(err)
        }
    }

    writer.Flush()

    if err := writer.Error(); err != nil {
        log.Fatal(err)
    }

    fmt.Println("Saved:", filename)
}

func main() {

    // 1. Load cleaned dataset
    file, err := os.Open("datasets/cleaned_algeria.csv")
    if err != nil {
        log.Fatal(err)
    }
    defer file.Close()

    records, err := csv.NewReader(file).ReadAll()
    if err != nil {
        log.Fatal(err)
    }

    if len(records) < 2 {
        log.Fatal("Dataset contains no observations.")
    }

    index := make(map[string]int)

    for i, name := range records[0] {
        index[strings.ToLower(strings.TrimSpace(name))] = i
    }

    numerical := []string{
        "Latitude", "Longitude", "Year",
    }

    categorical := []string{
        "Field Type", "Location",
    }

    target := "Flaring Vol (million m3)"

    required := append([]string{}, numerical...)
    required = append(required, categorical...)
    required = append(required, target)

    // Verify selected columns
    for _, name := range required {
        if _, ok := index[strings.ToLower(name)]; !ok {
            log.Fatalf("Missing column: %s", name)
        }
    }

    observations := []Observation{}

    // Function for safe numerical conversion
    parseNumber := func(value string) float64 {
        number, err := strconv.ParseFloat(value, 64)

        if err != nil || math.IsNaN(number) ||
            math.IsInf(number, 0) {
            log.Fatalf("Invalid numerical value: %s", value)
        }

        return number
    }

    // 2. Select predictor and target variables
    for _, row := range records[1:] {

        obs := Observation{}

        for _, name := range numerical {
            value := parseNumber(
                row[index[strings.ToLower(name)]])

            obs.Numerical = append(obs.Numerical, value)
        }

        for _, name := range categorical {
            obs.Categorical = append(
                obs.Categorical,
                row[index[strings.ToLower(name)]])
        }

        obs.Target = parseNumber(
            row[index[strings.ToLower(target)]])

        // Group repeated geographical coordinates
        obs.Group = fmt.Sprintf("%.4f_%.4f",
            obs.Numerical[0], obs.Numerical[1])

        observations = append(observations, obs)
    }

    // 3. Geographical group-based splitting
    groups := make(map[string][]Observation)

    for _, obs := range observations {
        groups[obs.Group] = append(groups[obs.Group], obs)
    }

    groupNames := []string{}

    for name := range groups {
        groupNames = append(groupNames, name)
    }

    // Sort before shuffling for reproducibility
    sort.Strings(groupNames)

    if len(groupNames) < 2 {
        log.Fatal("Not enough geographical groups for splitting.")
    }

    random := rand.New(rand.NewSource(42))

    random.Shuffle(len(groupNames), func(i, j int) {
        groupNames[i], groupNames[j] =
            groupNames[j], groupNames[i]
    })

    train := []Observation{}
    test := []Observation{}

    targetTrainSize := int(0.80 * float64(len(observations)))

    trainGroups := make(map[string]bool)
    testGroups := make(map[string]bool)

    for i, name := range groupNames {

        if len(train) < targetTrainSize &&
            i < len(groupNames)-1 {

            train = append(train, groups[name]...)
            trainGroups[name] = true

        } else {

            test = append(test, groups[name]...)
            testGroups[name] = true
        }
    }

    if len(train) == 0 || len(test) == 0 {
        log.Fatal("Training or testing dataset is empty.")
    }

    // Verify no geographical group overlap
    overlap := 0

    for name := range trainGroups {
        if testGroups[name] {
            overlap++
        }
    }

    // 4. Fit preprocessing using training data only
    prep := Preprocessor{
        NumericalNames: numerical,
        CategoricalNames: categorical,
    }

    // Calculate numerical scaling parameters
    for j := range numerical {

        values := []float64{}

        for _, obs := range train {
            values = append(values, obs.Numerical[j])
        }

        mean := stat.Mean(values, nil)
        sd := stat.StdDev(values, nil)

        if sd == 0 || math.IsNaN(sd) {
            sd = 1
        }

        prep.Means = append(prep.Means, mean)
        prep.StdDevs = append(prep.StdDevs, sd)
    }

    // Learn categorical levels from training data
    for j := range categorical {

        unique := make(map[string]bool)

        for _, obs := range train {
            unique[obs.Categorical[j]] = true
        }

        categories := []string{}

        for value := range unique {
            categories = append(categories, value)
        }

        sort.Strings(categories)

        prep.Categories = append(
            prep.Categories, categories)
    }

    // 5. Save processed datasets
    saveDataset("datasets/train_data.csv", train, prep)
    saveDataset("datasets/test_data.csv", test, prep)

    // Save preprocessing parameters for future prediction
    if err := os.MkdirAll("models", 0755); err != nil {
        log.Fatal(err)
    }

    parameters, err := json.MarshalIndent(prep, "", "  ")
    if err != nil {
        log.Fatal(err)
    }

    err = os.WriteFile(
        "models/preprocessing_parameters.json",
        parameters, 0644)

    if err != nil {
        log.Fatal(err)
    }

    // Calculate final feature count
    featureCount := len(numerical)

    for _, categories := range prep.Categories {
        featureCount += len(categories) + 1
    }

    // Display feature engineering summary
    fmt.Println("\n========== FEATURE SELECTION ==========")

    fmt.Println("Numerical Predictors:", numerical)
    fmt.Println("Categorical Predictors:", categorical)
    fmt.Println("Target Variable:", target)

    fmt.Println("Excluded Variables:",
        "COUNTRY, bcm, MMscfd, Flare Level")

    fmt.Println("Total Encoded Features:", featureCount)

    fmt.Println("\n========== NUMERICAL SCALING ==========")

    for j, name := range numerical {
        fmt.Printf("%s | Mean: %.4f | Std Dev: %.4f\n",
            name, prep.Means[j], prep.StdDevs[j])
    }

    fmt.Println("\n========== CATEGORICAL ENCODING ==========")

    for j, name := range categorical {
        fmt.Printf("%s: %d training categories\n",
            name, len(prep.Categories[j]))
    }

    fmt.Println("\n========== DATA SPLITTING SUMMARY ==========")

    total := float64(len(observations))

    fmt.Println("Total Observations:", len(observations))

    fmt.Printf("Training Rows: %d (%.2f%%)\n",
        len(train), float64(len(train))/total*100)

    fmt.Printf("Testing Rows: %d (%.2f%%)\n",
        len(test), float64(len(test))/total*100)

    fmt.Println("Total Geographical Groups:", len(groups))
    fmt.Println("Training Groups:", len(trainGroups))
    fmt.Println("Testing Groups:", len(testGroups))
    fmt.Println("Overlapping Groups:", overlap)
    fmt.Println("Random Seed: 42")

    fmt.Println("\n========== OUTPUT FILES ==========")
    fmt.Println("datasets/train_data.csv")
    fmt.Println("datasets/test_data.csv")
    fmt.Println("models/preprocessing_parameters.json")

    fmt.Println("\nFeature engineering completed successfully.")
}
