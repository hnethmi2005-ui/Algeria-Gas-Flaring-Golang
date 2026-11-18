
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
	"strings"

	"gonum.org/v1/gonum/stat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

// Numerical values used by Gonum Plot
func values(v []float64) plotter.Values {
	return plotter.Values(v)
}

// Calculate the median
func median(v []float64) float64 {
	c := append([]float64(nil), v...)
	sort.Float64s(c)
	n := len(c)
	if n%2 == 1 {
		return c[n/2]
	}
	return (c[n/2-1] + c[n/2]) / 2
}

// Save a chart as a PNG image
func save(p *plot.Plot, name string) {
	err := p.Save(9*vg.Inch, 5*vg.Inch, "figures/"+name+".png")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("Saved:", name+".png")
}

// Create a bar chart of average flaring volume
func barChart(groups map[string][]float64, title, filename string) {
	names := make([]string, 0, len(groups))
	for name := range groups {
		names = append(names, name)
	}
	sort.Strings(names)

	averages := make(plotter.Values, len(names))
	for i, name := range names {
		averages[i] = stat.Mean(groups[name], nil)
	}

	p := plot.New()
	p.Title.Text = title
	p.Y.Label.Text = "Average Flaring Volume (million m3)"

	bars, err := plotter.NewBarChart(averages, vg.Points(25))
	if err != nil {
		log.Fatal(err)
	}
	bars.Color = color.RGBA{R: 65, G: 105, B: 225, A: 255}
	p.Add(bars)
	p.NominalX(names...)

	save(p, filename)
}

// Correlation heatmap using coloured squares
func heatmap(names []string, columns map[string][]float64) {
	p := plot.New()
	p.Title.Text = "Numerical Variable Correlation Heatmap"
	p.X.Label.Text = "Variables"
	p.Y.Label.Text = "Variables"

	n := len(names)

	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			x := columns[names[i]]
			y := columns[names[j]]

			r := stat.Correlation(x, y, nil)

			// Undefined correlations are displayed in grey
			c := color.RGBA{R: 180, G: 180, B: 180, A: 255}

			if !math.IsNaN(r) {
				if r >= 0 {
					c = color.RGBA{
						R: uint8(255 * (1 - r)),
						G: uint8(255 * (1 - r)),
						B: 255,
						A: 255,
					}
				} else {
					c = color.RGBA{
						R: 255,
						G: uint8(255 * (1 + r)),
						B: uint8(255 * (1 + r)),
						A: 255,
					}
				}
			}

			square := plotter.XYs{
				{X: float64(j), Y: float64(i)},
				{X: float64(j+1), Y: float64(i)},
				{X: float64(j+1), Y: float64(i+1)},
				{X: float64(j), Y: float64(i+1)},
			}

			poly, err := plotter.NewPolygon(square)
			if err != nil {
				log.Fatal(err)
			}
			poly.Color = c
			poly.LineStyle.Width = 0
			p.Add(poly)
		}
	}

	p.X.Min, p.X.Max = 0, float64(n)
	p.Y.Min, p.Y.Max = 0, float64(n)

	// Numbered axes, with corresponding names printed below
	labels := make([]string, n)
	for i := range labels {
		labels[i] = strconv.Itoa(i + 1)
	}
	p.NominalX(labels...)
	p.NominalY(labels...)

	save(p, "correlation_heatmap")

	fmt.Println("\nCorrelation Heatmap Variable Key:")
	for i, name := range names {
		fmt.Printf("%d = %s\n", i+1, name)
	}

	fmt.Println("\nCorrelation Matrix:")
	for _, a := range names {
		fmt.Printf("%s:", a)
		for _, b := range names {
			fmt.Printf(" %.3f", stat.Correlation(
				columns[a], columns[b], nil))
		}
		fmt.Println()
	}
}

func main() {
	// Create output folder
	if err := os.MkdirAll("figures", 0755); err != nil {
		log.Fatal(err)
	}

	// Load cleaned dataset
	file, err := os.Open("datasets/cleaned_algeria.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	headers := records[0]
	data := records[1:]

	index := make(map[string]int)
	for i, h := range headers {
		index[strings.ToLower(strings.TrimSpace(h))] = i
	}

	// Confirm required columns
	required := []string{
		"latitude", "longitude", "bcm", "mmscfd",
		"year", "flaring vol (million m3)",
		"field type", "flare level",
	}
	for _, name := range required {
		if _, ok := index[name]; !ok {
			log.Fatalf("Missing column: %s", name)
		}
	}

	numericNames := []string{
		"Latitude", "Longitude", "bcm",
		"MMscfd", "Year", "Flaring Vol (million m3)",
	}

	columns := make(map[string][]float64)
	fieldGroups := make(map[string][]float64)
	levelGroups := make(map[string][]float64)
	yearGroups := make(map[int][]float64)

	for _, row := range data {
		for _, name := range numericNames {
			key := strings.ToLower(name)
			value, err := strconv.ParseFloat(row[index[key]], 64)
			if err != nil {
				log.Fatalf("Invalid numerical value in %s: %v", name, err)
			}
			columns[name] = append(columns[name], value)
		}

		flaring, _ := strconv.ParseFloat(
			row[index["flaring vol (million m3)"]], 64)

		year, _ := strconv.Atoi(row[index["year"]])

		field := row[index["field type"]]
		level := row[index["flare level"]]

		fieldGroups[field] = append(fieldGroups[field], flaring)
		levelGroups[level] = append(levelGroups[level], flaring)
		yearGroups[year] = append(yearGroups[year], flaring)
	}

	// 1. Descriptive statistics
	fmt.Println("========== DESCRIPTIVE STATISTICS ==========")
	fmt.Printf("%-27s %12s %12s %12s %12s %12s\n",
		"Variable", "Mean", "Median", "Std Dev", "Min", "Max")

	for _, name := range numericNames {
		v := columns[name]
		min, max := v[0], v[0]

		for _, x := range v {
			if x < min {
				min = x
			}
			if x > max {
				max = x
			}
		}

		fmt.Printf("%-27s %12.3f %12.3f %12.3f %12.3f %12.3f\n",
			name, stat.Mean(v, nil), median(v),
			stat.StdDev(v, nil), min, max)
	}

	flaring := columns["Flaring Vol (million m3)"]

	// 2. Histogram
	p1 := plot.New()
	p1.Title.Text = "Distribution of Gas Flaring Volume"
	p1.X.Label.Text = "Flaring Volume (million m3)"
	p1.Y.Label.Text = "Frequency"

	hist, err := plotter.NewHist(values(flaring), 30)
	if err != nil {
		log.Fatal(err)
	}
	hist.FillColor = color.RGBA{R: 70, G: 130, B: 180, A: 255}
	p1.Add(hist)
	save(p1, "flaring_histogram")

	// 3. Box plot
	p2 := plot.New()
	p2.Title.Text = "Box Plot of Gas Flaring Volume"
	p2.Y.Label.Text = "Flaring Volume (million m3)"

	box, err := plotter.NewBoxPlot(vg.Points(50), 0, values(flaring))
	if err != nil {
		log.Fatal(err)
	}
	p2.Add(box)
	p2.NominalX("Algeria")
	save(p2, "flaring_boxplot")

	// 4. Annual flaring trend
	years := make([]int, 0, len(yearGroups))
	for year := range yearGroups {
		years = append(years, year)
	}
	sort.Ints(years)

	points := make(plotter.XYs, len(years))

	fmt.Println("\n========== ANNUAL FLARING VOLUME ==========")

	for i, year := range years {
		total := 0.0
		for _, v := range yearGroups[year] {
			total += v
		}
		points[i].X = float64(year)
		points[i].Y = total
		fmt.Printf("%d: %.3f million m3\n", year, total)
	}

	p3 := plot.New()
	p3.Title.Text = "Annual Total Gas Flaring Volume"
	p3.X.Label.Text = "Year"
	p3.Y.Label.Text = "Total Flaring Volume (million m3)"

	line, err := plotter.NewLine(points)
	if err != nil {
		log.Fatal(err)
	}
	line.Color = color.RGBA{R: 220, G: 70, B: 70, A: 255}
	line.Width = vg.Points(2)
	p3.Add(line)
	save(p3, "annual_flaring_trend")

	// 5. Category comparisons
	barChart(fieldGroups,
		"Average Flaring Volume by Field Type",
		"field_type_comparison")

	barChart(levelGroups,
		"Average Flaring Volume by Flare Level",
		"flare_level_comparison")

	// 6. Correlation heatmap
	heatmap(numericNames, columns)

	fmt.Println("\nEDA completed successfully.")
	fmt.Println("All six figures saved in the figures directory.")
}
