
package main

import (
    "encoding/csv"
    "fmt"
    "os"
    "strings"
    "strconv"
    "sort"
    "math"
    "log"
)

// Check missing values
func isMissing(value string) bool {
    value = strings.ToLower(strings.TrimSpace(value))

    return value == "" || value == "na" ||
        value == "n/a" || value == "null" ||
        value == "nan" || value == "none"
}

// Calculate percentile for IQR
func percentile(values []float64, p float64) float64 {

    position := float64(len(values)-1) * p
    lower := int(math.Floor(position))
    upper := int(math.Ceil(position))

    if lower == upper {
        return values[lower]
    }

    return values[lower] +
        (position-float64(lower)) *
        (values[upper]-values[lower])
}

func main() {

    // Load original dataset
    file, err := os.Open("datasets/NGDS_CO2_Algeria.csv")
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

    headers := records[0]
    data := records[1:]

    // Standardise column names
    columns := make(map[string]int)

    for i, col := range headers {
        headers[i] = strings.Join(
            strings.Fields(strings.TrimPrefix(col, "\uFEFF")), " ")

        columns[strings.ToLower(headers[i])] = i
    }

    numerical := []string{
        "latitude", "longitude", "bcm",
        "mmscfd", "year", "flaring vol (million m3)",
    }

    categorical := []string{
        "country", "field type", "location", "flare level",
    }

    // Verify required columns
    for _, col := range append(numerical, categorical...) {
        if _, exists := columns[col]; !exists {
            log.Fatalf("Required column not found: %s", col)
        }
    }

    target := columns["flaring vol (million m3)"]

    cleaned := [][]string{headers}
    seen := make(map[string]bool)
    flaringValues := []float64{}

    missingReplaced := 0
    duplicates := 0
    invalidRows := 0
    invalidCoordinates := 0

    // Clean each observation
    for _, row := range data {

        // Remove unnecessary spaces
        for j := range row {
            row[j] = strings.TrimSpace(row[j])
        }

        replaced := 0

        // Handle missing categorical values
        // and standardise categorical text
        for _, col := range categorical {

            index := columns[col]

            if isMissing(row[index]) {
                row[index] = "Unknown"
                replaced++
            } else {
                row[index] = strings.ToUpper(
                    strings.Join(strings.Fields(row[index]), " "))
            }
        }

        // Validate numerical values
        valid := true

        for _, col := range numerical {

            index := columns[col]

            value, err := strconv.ParseFloat(row[index], 64)

            if err != nil || math.IsNaN(value) ||
                math.IsInf(value, 0) {

                valid = false
                break
            }

            // Flaring measurements cannot be negative
            if (col == "bcm" || col == "mmscfd" ||
                col == "flaring vol (million m3)") &&
                value < 0 {

                valid = false
                break
            }

            // Year must be a positive whole number
            if col == "year" &&
                (value <= 0 || value != math.Trunc(value)) {

                valid = false
                break
            }
        }

        if !valid {
            invalidRows++
            continue
        }

        // Validate latitude and longitude
        lat, _ := strconv.ParseFloat(
            row[columns["latitude"]], 64)

        lon, _ := strconv.ParseFloat(
            row[columns["longitude"]], 64)

        if lat < -90 || lat > 90 ||
            lon < -180 || lon > 180 {

            invalidCoordinates++
            continue
        }

        // Check duplicate records
        key := strings.Join(row, "\x00")

        if seen[key] {
            duplicates++
            continue
        }

        seen[key] = true

        missingReplaced += replaced

        // Store valid observations
        cleaned = append(cleaned, row)

        value, _ := strconv.ParseFloat(row[target], 64)
        flaringValues = append(flaringValues, value)
    }

    if len(flaringValues) == 0 {
        log.Fatal("No valid observations remain after cleaning.")
    }

    // Detect outliers using the IQR method
    sort.Float64s(flaringValues)

    q1 := percentile(flaringValues, 0.25)
    q3 := percentile(flaringValues, 0.75)

    iqr := q3 - q1

    lowerBound := q1 - 1.5*iqr
    upperBound := q3 + 1.5*iqr

    outliers := 0

    for _, value := range flaringValues {
        if value < lowerBound || value > upperBound {
            outliers++
        }
    }

    // Verify remaining missing values
    remainingMissing := 0

    for _, row := range cleaned[1:] {
        for _, value := range row {
            if isMissing(value) {
                remainingMissing++
            }
        }
    }

    // Save cleaned dataset
    output, err := os.Create("datasets/cleaned_algeria.csv")
    if err != nil {
        log.Fatal(err)
    }

    writer := csv.NewWriter(output)
    writer.WriteAll(cleaned)

    if err := writer.Error(); err != nil {
        output.Close()
        log.Fatal(err)
    }

    if err := output.Close(); err != nil {
        log.Fatal(err)
    }

    // Display cleaning summary
    fmt.Println("========== DATA CLEANING SUMMARY ==========")

    fmt.Println("Original Rows:", len(data))
    fmt.Println("Cleaned Rows:", len(cleaned)-1)
    fmt.Println("Total Columns:", len(headers))
    fmt.Println("Missing Values Replaced:", missingReplaced)
    fmt.Println("Duplicates Removed:", duplicates)
    fmt.Println("Invalid Numerical Rows Removed:", invalidRows)
    fmt.Println("Invalid Coordinate Rows Removed:", invalidCoordinates)
    fmt.Println("Remaining Missing Values:", remainingMissing)

    // Display IQR results
    fmt.Println("\n========== IQR OUTLIER ANALYSIS ==========")

    fmt.Printf("Q1: %.4f\n", q1)
    fmt.Printf("Q3: %.4f\n", q3)
    fmt.Printf("IQR: %.4f\n", iqr)
    fmt.Printf("Lower Bound: %.4f\n", lowerBound)
    fmt.Printf("Upper Bound: %.4f\n", upperBound)

    fmt.Println("Potential Outliers:", outliers)
    fmt.Println("Outliers Removed: 0")

    fmt.Println("\n========== DATASET SAVED ==========")
    fmt.Println("File: datasets/cleaned_algeria.csv")
    fmt.Println("Data preprocessing completed successfully.")
}
