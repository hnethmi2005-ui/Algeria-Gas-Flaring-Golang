
package main

import (
    "fmt"
    "runtime"

    "github.com/go-gota/gota/dataframe"
    "gonum.org/v1/gonum/stat"
    "gonum.org/v1/gonum/mat"
    "gonum.org/v1/plot"
    "github.com/go-echarts/go-echarts/v2/charts"
    "github.com/paulmach/orb"
)

func main() {

    fmt.Println("========================================")
    fmt.Println("ALGERIA GAS FLARING ANALYSIS PROJECT")
    fmt.Println("Golang Environment Test")
    fmt.Println("========================================")

    // Display the running Go version
    fmt.Println("Go Version:", runtime.Version())

    // Test Gota using synthetic data
    df := dataframe.LoadRecords([][]string{
        {"ID", "TestValue"},
        {"1", "10"},
        {"2", "20"},
        {"3", "30"},
    })

    fmt.Println("\nGota DataFrame Test:")
    fmt.Println(df)

    // Test Gonum statistical calculations
    values := []float64{10, 20, 30, 40, 50}

    mean := stat.Mean(values, nil)
    stdDev := stat.StdDev(values, nil)

    fmt.Printf("\nMean: %.2f\n", mean)
    fmt.Printf("Standard Deviation: %.2f\n", stdDev)

    // Test Gonum matrix operations
    matrix := mat.NewDense(2, 2, []float64{
        1, 2,
        3, 4,
    })

    fmt.Println("\nGonum Matrix Test:")
    fmt.Println(mat.Formatted(matrix))

    // Test Gonum Plot
    p := plot.New()
    p.Title.Text = "Environment Test Plot"

    fmt.Println("\nGonum Plot:", p.Title.Text)

    // Test Go-ECharts
    chart := charts.NewBar()

    fmt.Printf(
        "Go-ECharts initialised: %t\n",
        chart != nil,
    )

    // Test Orb using synthetic coordinates
    point := orb.Point{0.0, 0.0}

    fmt.Println("Orb Geographic Point:", point)

    fmt.Println("\n========================================")
    fmt.Println("All library tests completed successfully.")
    fmt.Println("Golang environment is ready.")
    fmt.Println("========================================")
}
