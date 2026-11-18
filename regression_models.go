
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
    "time"

    "gonum.org/v1/gonum/mat"
)

// Dataset structure
type Dataset struct {
    X [][]float64
    Y []float64
    Features []string
}

// Linear Regression model
type LinearModel struct {
    Coefficients []float64
}

// Decision Tree node
type TreeNode struct {
    Feature int
    Threshold float64
    Value float64
    Left *TreeNode
    Right *TreeNode
}

// Random Forest model
type ForestModel struct {
    Trees []*TreeNode
}

// Load processed CSV dataset
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

    headers := records[0]
    last := len(headers)-1

    if headers[last] != "Flaring Vol (million m3)" {
        log.Fatal("Unexpected target column.")
    }

    data := Dataset{
        Features: headers[:last],
    }

    for _, row := range records[1:] {

        values := make([]float64, len(row))

        for j, text := range row {
            value, err := strconv.ParseFloat(text, 64)

            if err != nil || math.IsNaN(value) ||
                math.IsInf(value, 0) {
                log.Fatal("Invalid numerical value:", text)
            }

            values[j] = value
        }

        data.X = append(data.X, values[:last])
        data.Y = append(data.Y, values[last])
    }

    return data
}

// Multiple Linear Regression using SVD
func trainLinear(data Dataset) LinearModel {

    n := len(data.X)
    p := len(data.Features)

    // Add intercept column
    X := mat.NewDense(n, p+1, nil)

    for i := 0; i < n; i++ {

        X.Set(i, 0, 1)

        for j := 0; j < p; j++ {
            X.Set(i, j+1, data.X[i][j])
        }
    }

    Y := mat.NewDense(n, 1, data.Y)

    // SVD handles redundant and constant features
    var svd mat.SVD

    if !svd.Factorize(X, mat.SVDThin) {
        log.Fatal("SVD factorisation failed.")
    }

    rank := svd.Rank(1e-10)

    if rank == 0 {
        log.Fatal("Invalid matrix rank.")
    }

    var coefficients mat.Dense

    svd.SolveTo(&coefficients, Y, rank)

    result := make([]float64, p+1)

    for j := range result {
        result[j] = coefficients.At(j, 0)
    }

    fmt.Println("Linear Regression matrix rank:", rank)

    return LinearModel{Coefficients: result}
}

// Linear Regression prediction
func predictLinear(model LinearModel, x []float64) float64 {

    prediction := model.Coefficients[0]

    for j, value := range x {
        prediction += model.Coefficients[j+1]*value
    }

    return prediction
}

// Calculate mean target value
func meanTarget(y []float64, ids []int) float64 {

    total := 0.0

    for _, id := range ids {
        total += y[id]
    }

    return total / float64(len(ids))
}

// Calculate sum of squared errors
func squaredError(y []float64, ids []int) float64 {

    mean := meanTarget(y, ids)
    errorSum := 0.0

    for _, id := range ids {
        difference := y[id]-mean
        errorSum += difference*difference
    }

    return errorSum
}

// Build a regression decision tree
func buildTree(
    X [][]float64,
    y []float64,
    ids []int,
    depth int,
    maxDepth int,
    minLeaf int,
    maxFeatures int,
    rng *rand.Rand,
) *TreeNode {

    node := &TreeNode{
        Feature: -1,
        Value: meanTarget(y, ids),
    }

    // Stop growing the tree
    if depth >= maxDepth ||
        len(ids) < 2*minLeaf {

        return node
    }

    p := len(X[0])

    features := rng.Perm(p)

    if maxFeatures < p {
        features = features[:maxFeatures]
    }

    bestError := squaredError(y, ids)
    bestFeature := -1
    bestThreshold := 0.0

    // Find the best feature and splitting threshold
    for _, feature := range features {

        sortedIDs := append([]int(nil), ids...)

        sort.Slice(sortedIDs, func(i, j int) bool {
            return X[sortedIDs[i]][feature] <
                   X[sortedIDs[j]][feature]
        })

        totalY := 0.0
        totalY2 := 0.0

        for _, id := range sortedIDs {
            totalY += y[id]
            totalY2 += y[id]*y[id]
        }

        leftSum := 0.0
        leftSum2 := 0.0

        for i := 0; i < len(sortedIDs)-1; i++ {

            id := sortedIDs[i]

            leftSum += y[id]
            leftSum2 += y[id]*y[id]

            leftCount := i+1
            rightCount := len(sortedIDs)-leftCount

            if leftCount < minLeaf ||
                rightCount < minLeaf {
                continue
            }

            a := X[sortedIDs[i]][feature]
            b := X[sortedIDs[i+1]][feature]

            if a == b {
                continue
            }

            rightSum := totalY-leftSum
            rightSum2 := totalY2-leftSum2

            leftError := leftSum2 -
                leftSum*leftSum/float64(leftCount)

            rightError := rightSum2 -
                rightSum*rightSum/float64(rightCount)

            splitError := math.Max(0, leftError) +
                math.Max(0, rightError)

            if splitError < bestError-1e-9 {

                bestError = splitError
                bestFeature = feature
                bestThreshold = (a+b)/2
            }
        }
    }

    // No useful split found
    if bestFeature == -1 {
        return node
    }

    leftIDs := []int{}
    rightIDs := []int{}

    for _, id := range ids {

        if X[id][bestFeature] <= bestThreshold {
            leftIDs = append(leftIDs, id)
        } else {
            rightIDs = append(rightIDs, id)
        }
    }

    if len(leftIDs) < minLeaf ||
        len(rightIDs) < minLeaf {
        return node
    }

    node.Feature = bestFeature
    node.Threshold = bestThreshold

    node.Left = buildTree(
        X, y, leftIDs, depth+1,
        maxDepth, minLeaf, maxFeatures, rng,
    )

    node.Right = buildTree(
        X, y, rightIDs, depth+1,
        maxDepth, minLeaf, maxFeatures, rng,
    )

    return node
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

// Train Random Forest Regression
func trainForest(
    data Dataset,
    numberTrees int,
    maxDepth int,
    minLeaf int,
) ForestModel {

    rng := rand.New(rand.NewSource(42))

    n := len(data.X)

    maxFeatures := int(
        math.Sqrt(float64(len(data.Features))),
    )

    if maxFeatures < 1 {
        maxFeatures = 1
    }

    forest := ForestModel{}

    for t := 0; t < numberTrees; t++ {

        // Bootstrap sampling with replacement
        ids := make([]int, n)

        for i := range ids {
            ids[i] = rng.Intn(n)
        }

        tree := buildTree(
            data.X,
            data.Y,
            ids,
            0,
            maxDepth,
            minLeaf,
            maxFeatures,
            rng,
        )

        forest.Trees = append(forest.Trees, tree)

        if (t+1)%20 == 0 {
            fmt.Printf("Trees trained: %d/%d\n",
                t+1, numberTrees)
        }
    }

    return forest
}

// Random Forest prediction
func predictForest(model ForestModel, x []float64) float64 {

    total := 0.0

    for _, tree := range model.Trees {
        total += predictTree(tree, x)
    }

    return total/float64(len(model.Trees))
}

// Save trained models as JSON
func saveModel(path string, model interface{}) {

    content, err := json.MarshalIndent(model, "", "  ")
    if err != nil {
        log.Fatal(err)
    }

    if err := os.WriteFile(path, content, 0644); err != nil {
        log.Fatal(err)
    }

    fmt.Println("Saved:", path)
}

// Save actual and predicted values
func savePredictions(
    path string,
    actual, linear, tree, forest []float64,
) {

    file, err := os.Create(path)
    if err != nil {
        log.Fatal(err)
    }
    defer file.Close()

    writer := csv.NewWriter(file)

    writer.Write([]string{
        "Actual",
        "Linear_Regression",
        "Decision_Tree",
        "Random_Forest",
    })

    for i := range actual {

        row := []string{
            strconv.FormatFloat(actual[i], 'g', -1, 64),
            strconv.FormatFloat(linear[i], 'g', -1, 64),
            strconv.FormatFloat(tree[i], 'g', -1, 64),
            strconv.FormatFloat(forest[i], 'g', -1, 64),
        }

        if err := writer.Write(row); err != nil {
            log.Fatal(err)
        }
    }

    writer.Flush()

    if err := writer.Error(); err != nil {
        log.Fatal(err)
    }

    fmt.Println("Saved:", path)
}

func main() {

    // Create output directories
    for _, dir := range []string{"models", "outputs"} {
        if err := os.MkdirAll(dir, 0755); err != nil {
            log.Fatal(err)
        }
    }

    // Load training and testing datasets
    train := loadData("datasets/train_data.csv")
    test := loadData("datasets/test_data.csv")

    if strings.Join(train.Features, "\x00") !=
       strings.Join(test.Features, "\x00") {
        log.Fatal("Training and testing features do not match.")
    }

    fmt.Println("========== DATASET SUMMARY ==========")

    fmt.Println("Training Observations:", len(train.X))
    fmt.Println("Testing Observations:", len(test.X))
    fmt.Println("Encoded Features:", len(train.Features))

    // Train Multiple Linear Regression
    fmt.Println("\n========== LINEAR REGRESSION ==========")

    start := time.Now()

    linear := trainLinear(train)

    fmt.Println("Training Time:", time.Since(start))
    fmt.Println("Linear Regression training completed.")

    // Train Decision Tree Regression
    fmt.Println("\n========== DECISION TREE ==========")

    start = time.Now()

    ids := make([]int, len(train.X))

    for i := range ids {
        ids[i] = i
    }

    rng := rand.New(rand.NewSource(42))

    tree := buildTree(
        train.X,
        train.Y,
        ids,
        0,
        8,
        10,
        len(train.Features),
        rng,
    )

    fmt.Println("Training Time:", time.Since(start))
    fmt.Println("Decision Tree training completed.")

    // Train Random Forest Regression
    fmt.Println("\n========== RANDOM FOREST ==========")

    start = time.Now()

    forest := trainForest(train, 100, 10, 5)

    fmt.Println("Training Time:", time.Since(start))
    fmt.Println("Random Forest training completed.")

    // Generate testing predictions
    linearPred := make([]float64, len(test.X))
    treePred := make([]float64, len(test.X))
    forestPred := make([]float64, len(test.X))

    for i, x := range test.X {

        linearPred[i] = predictLinear(linear, x)
        treePred[i] = predictTree(tree, x)
        forestPred[i] = predictForest(forest, x)
    }

    // Save trained models
    fmt.Println("\n========== SAVING MODELS ==========")

    saveModel("models/linear_regression.json", linear)
    saveModel("models/decision_tree.json", tree)
    saveModel("models/random_forest.json", forest)
    saveModel("models/feature_names.json", train.Features)

    // Save testing predictions
    savePredictions(
        "outputs/test_predictions.csv",
        test.Y,
        linearPred,
        treePred,
        forestPred,
    )

    // Display first 10 predictions
    fmt.Println("\n========== SAMPLE PREDICTIONS ==========")

    fmt.Printf("%-8s %12s %12s %12s %12s\n",
        "Record", "Actual", "Linear", "Tree", "Forest")

    for i := 0; i < len(test.Y) && i < 10; i++ {

        fmt.Printf("%-8d %12.3f %12.3f %12.3f %12.3f\n",
            i+1,
            test.Y[i],
            linearPred[i],
            treePred[i],
            forestPred[i],
        )
    }

    fmt.Println("\n========== TRAINING SUMMARY ==========")
    fmt.Println("Total Models Trained: 3")
    fmt.Println("Random Forest Trees:", len(forest.Trees))
    fmt.Println("Random Seed: 42")
    fmt.Println("Testing Predictions:", len(test.Y))

    fmt.Println("\nModel development completed successfully.")
}
