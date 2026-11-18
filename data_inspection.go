
package main

import (
    "encoding/csv"
    "fmt"
    "os"
    "strings"
    "strconv"
    "math"
    "log"
)

// Identify missing values
func isMissing(value string) bool {
    value = strings.ToLower(strings.TrimSpace(value))

    return value == "" || value == "na" ||
        value == "n/a" || value == "null" ||
        value == "nan" || value == "none"
}

func main() {

    // Load the CSV dataset
    file, err := os.Open("datasets/NGDS_CO2_Algeria.csv")
    if err != nil {
        log.Fatal(err)
    }
    defer file.Close()

    reader := csv.NewReader(file)
    records, err := reader.ReadAll()

    if err != nil {
        log.Fatal(err)
    }

    if len(records) < 2 {
        log.Fatal("Dataset contains no observations.")
    }

    headers := records[0]
    headers[0] = strings.TrimPrefix(headers[0], "\uFEFF")
    data := records[1:]

    // 1. Dataset dimensions
    fmt.Println("========== DATASET DIMENSIONS ==========")
    fmt.Println("Total Rows:", len(data))
    fmt.Println("Total Columns:", len(headers))

    // 2. Column names
    fmt.Println("\n========== COLUMN NAMES ==========")

    for i, col := range headers {
        fmt.Printf("%d. %s\n", i+1, col)
    }

    // 3. Display the first 10 observations
    fmt.Println("\n========== FIRST 10 OBSERVATIONS ==========")
    fmt.Println(strings.Join(headers, " | "))

    for i, row := range data {
        if i >= 10 {
            break
        }
        fmt.Println(strings.Join(row, " | "))
    }

    // 4. Identify data types and missing values
    fmt.Println("\n========== DATA TYPES AND MISSING VALUES ==========")

    numerical := []string{}
    categorical := []string{}

    totalMissing := 0

    fmt.Printf("%-30s %-18s %s\n", "Column", "Data Type", "Missing")

    for j, col := range headers {

        missing := 0
        numeric := 0
        nonMissing := 0

        for _, row := range data {

            value := strings.TrimSpace(row[j])

            if isMissing(value) {
                missing++
                continue
            }

            nonMissing++

            number, err := strconv.ParseFloat(value, 64)

            if err == nil && !math.IsNaN(number) && !math.IsInf(number, 0) {
                numeric++
            }
        }

        dataType := "Categorical/Text"

        if nonMissing == 0 {
            dataType = "Unknown"
        } else if numeric == nonMissing {
            dataType = "Numerical"
            numerical = append(numerical, col)
        } else if numeric > 0 {
            dataType = "Mixed"
        } else {
            categorical = append(categorical, col)
        }

        totalMissing += missing

        fmt.Printf("%-30s %-18s %d\n", col, dataType, missing)
    }

    fmt.Println("\nNumerical Variables:", numerical)
    fmt.Println("Categorical/Text Variables:", categorical)
    fmt.Println("Total Missing Values:", totalMissing)

    // 5. Identify duplicate records
    fmt.Println("\n========== DUPLICATE RECORDS ==========")

    seen := make(map[string]bool)
    duplicates := 0

    for _, row := range data {

        key := strings.Join(row, "\x00")

        if seen[key] {
            duplicates++
        } else {
            seen[key] = true
        }
    }

    fmt.Println("Duplicate Records:", duplicates)

    // 6. Identify potential regression target variables
    fmt.Println("\n========== POTENTIAL TARGET VARIABLES ==========")

    targets := []string{}

    for _, col := range numerical {

        name := strings.ToLower(col)

        if strings.Contains(name, "co2") ||
            strings.Contains(name, "co₂") ||
            strings.Contains(name, "emission") ||
            strings.Contains(name, "volume") ||
            strings.Contains(name, "flare") ||
            strings.Contains(name, "flaring") {

            targets = append(targets, col)
        }
    }

    fmt.Println("Potential Regression Targets:", targets)

    // 7. Identify geographical variables
    fmt.Println("\n========== GEOGRAPHICAL VARIABLES ==========")

    geographical := []string{}

    for _, col := range headers {

        name := strings.ToLower(strings.TrimSpace(col))

        if strings.Contains(name, "latitude") ||
            strings.Contains(name, "longitude") ||
            strings.Contains(name, "location") ||
            strings.Contains(name, "region") ||
            strings.Contains(name, "coordinate") ||
            name == "lat" || name == "lon" ||
            name == "lng" || name == "country" {

            geographical = append(geographical, col)
        }
    }

    fmt.Println("Potential Geographical Variables:", geographical)

    fmt.Println("\n========== INSPECTION COMPLETED ==========")
}
