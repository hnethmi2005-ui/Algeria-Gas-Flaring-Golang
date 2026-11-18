
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

    "github.com/paulmach/orb"
    "gonum.org/v1/plot"
    "gonum.org/v1/plot/plotter"
    "gonum.org/v1/plot/vg"
)

// Store geographical observations
type Observation struct {
    Point     orb.Point
    Volume    float64
    FieldType string
    Location  string
}

// Store location-level results
type Hotspot struct {
    Name   string
    Total  float64
    Count  int
}

// Create a geographical plot
func createPlot(title string) *plot.Plot {
    p := plot.New()
    p.Title.Text = title
    p.X.Label.Text = "Longitude"
    p.Y.Label.Text = "Latitude"
    p.Add(plotter.NewGrid())
    return p
}

// Save geographical figures
func save(p *plot.Plot, filename string) {
    err := p.Save(10*vg.Inch, 7*vg.Inch,
        "maps/"+filename+".png")

    if err != nil {
        log.Fatal(err)
    }

    fmt.Println("Saved:", filename+".png")
}

func main() {

    // Create maps directory
    if err := os.MkdirAll("maps", 0755); err != nil {
        log.Fatal(err)
    }

    // Load cleaned dataset
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

    // Identify column positions
    index := make(map[string]int)

    for i, name := range records[0] {
        index[strings.ToLower(strings.TrimSpace(name))] = i
    }

    required := []string{
        "latitude", "longitude", "field type",
        "location", "flaring vol (million m3)",
    }

    for _, name := range required {
        if _, ok := index[name]; !ok {
            log.Fatalf("Column not found: %s", name)
        }
    }

    observations := []Observation{}
    totalVolume := 0.0
    maxVolume := 0.0

    // Extract geographical variables
    for _, row := range records[1:] {

        lat, e1 := strconv.ParseFloat(
            row[index["latitude"]], 64)

        lon, e2 := strconv.ParseFloat(
            row[index["longitude"]], 64)

        volume, e3 := strconv.ParseFloat(
            row[index["flaring vol (million m3)"]], 64)

        if e1 != nil || e2 != nil || e3 != nil ||
            math.IsNaN(lat) || math.IsNaN(lon) ||
            math.IsNaN(volume) || math.IsInf(volume, 0) ||
            lat < -90 || lat > 90 ||
            lon < -180 || lon > 180 || volume < 0 {
            continue
        }

        observations = append(observations, Observation{
            Point:     orb.Point{lon, lat},
            Volume:    volume,
            FieldType: row[index["field type"]],
            Location:  row[index["location"]],
        })

        totalVolume += volume
        maxVolume = math.Max(maxVolume, volume)
    }

    if len(observations) == 0 {
        log.Fatal("No valid geographical observations.")
    }

    // 1. Geographical distribution
    p1 := createPlot("Geographical Distribution of Gas Flaring")

    points := make(plotter.XYs, len(observations))

    for i, obs := range observations {
        points[i].X = obs.Point[0]
        points[i].Y = obs.Point[1]
    }

    scatter, err := plotter.NewScatter(points)
    if err != nil {
        log.Fatal(err)
    }

    scatter.GlyphStyle.Color = color.RGBA{30, 100, 200, 130}
    scatter.GlyphStyle.Radius = vg.Points(2.5)

    p1.Add(scatter)
    save(p1, "geographical_distribution")

    // 2. Flaring intensity visualisation
    p2 := createPlot("Gas Flaring Intensity in Algeria")

    // Draw lower-volume observations first
    sort.Slice(observations, func(i, j int) bool {
        return observations[i].Volume < observations[j].Volume
    })

    for _, obs := range observations {

        ratio := 0.0
        if maxVolume > 0 {
            ratio = obs.Volume / maxVolume
        }

        point := plotter.XYs{
            {X: obs.Point[0], Y: obs.Point[1]},
        }

        s, err := plotter.NewScatter(point)
        if err != nil {
            log.Fatal(err)
        }

        // Blue = lower volume, red = higher volume
        s.GlyphStyle.Color = color.RGBA{
            R: uint8(255 * ratio),
            G: 60,
            B: uint8(255 * (1-ratio)),
            A: 180,
        }

        // Larger markers indicate higher volume
        s.GlyphStyle.Radius = vg.Points(
            2 + 10*math.Sqrt(ratio))

        p2.Add(s)
    }

    save(p2, "flaring_intensity_hotspots")

    // 3. Compare geographical Field Type patterns
    p3 := createPlot("Geographical Distribution by Field Type")

    fieldGroups := make(map[string]plotter.XYs)

    for _, obs := range observations {

        fieldGroups[obs.FieldType] = append(
            fieldGroups[obs.FieldType],
            plotter.XY{
                X: obs.Point[0],
                Y: obs.Point[1],
            })
    }

    colours := []color.Color{
        color.RGBA{30, 144, 255, 200},
        color.RGBA{255, 140, 0, 200},
        color.RGBA{34, 139, 34, 200},
        color.RGBA{128, 128, 128, 200},
    }

    fields := []string{}

    for field := range fieldGroups {
        fields = append(fields, field)
    }

    sort.Strings(fields)

    for i, field := range fields {

        s, err := plotter.NewScatter(fieldGroups[field])
        if err != nil {
            log.Fatal(err)
        }

        s.GlyphStyle.Color = colours[i%len(colours)]
        s.GlyphStyle.Radius = vg.Points(3)

        p3.Add(s)
        p3.Legend.Add(field, s)
    }

    p3.Legend.Top = true

    save(p3, "geographical_field_type")

    // 4. Identify high-volume locations
    locationTotals := make(map[string]float64)
    locationCounts := make(map[string]int)

    for _, obs := range observations {

        location := obs.Location

        if strings.EqualFold(location, "Unknown") ||
            strings.TrimSpace(location) == "" {
            continue
        }

        locationTotals[location] += obs.Volume
        locationCounts[location]++
    }

    hotspots := []Hotspot{}

    for location, total := range locationTotals {

        hotspots = append(hotspots, Hotspot{
            Name:  location,
            Total: total,
            Count: locationCounts[location],
        })
    }

    // Rank locations by total recorded volume
    sort.Slice(hotspots, func(i, j int) bool {
        if hotspots[i].Total == hotspots[j].Total {
            return hotspots[i].Name < hotspots[j].Name
        }
        return hotspots[i].Total > hotspots[j].Total
    })

    fmt.Println("\n========== TOP 10 FLARING LOCATIONS ==========")

    fmt.Printf("%-5s %-30s %15s %10s\n",
        "Rank", "Location", "Total Volume", "Records")

    for i, h := range hotspots {

        if i >= 10 {
            break
        }

        fmt.Printf("%-5d %-30s %15.2f %10d\n",
            i+1, h.Name, h.Total, h.Count)
    }

    // 5. Field Type spatial summary
    fmt.Println("\n========== FIELD TYPE SUMMARY ==========")

    for _, field := range fields {

        total := 0.0
        count := 0

        for _, obs := range observations {
            if obs.FieldType == field {
                total += obs.Volume
                count++
            }
        }

        fmt.Printf(
            "%s: %d records | Total: %.2f | Average: %.2f million m3\n",
            field, count, total, total/float64(count))
    }

    // Geographical summary
    fmt.Println("\n========== SPATIAL ANALYSIS SUMMARY ==========")

    fmt.Println("Valid Observations:", len(observations))
    fmt.Println("Skipped Observations:",
        len(records)-1-len(observations))

    fmt.Println("Unique Known Location Labels:",
        len(locationTotals))

    fmt.Printf("Total Recorded Volume: %.2f million m3\n",
        totalVolume)

    fmt.Printf("Maximum Individual Volume: %.2f million m3\n",
        maxVolume)

    if len(hotspots) > 0 {
        fmt.Println("Highest Total Volume Location:",
            hotspots[0].Name)

        fmt.Printf("Location Total: %.2f million m3\n",
            hotspots[0].Total)
    }

    fmt.Println("\nGeographical analysis completed successfully.")
    fmt.Println("All three figures saved in the maps directory.")
}
