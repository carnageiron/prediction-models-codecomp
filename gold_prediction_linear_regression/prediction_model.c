#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define MAX_DATA_POINTS 1000

void normalize(double *data, int n, double *min_val, double *max_val) {
    *min_val = data[0];
    *max_val = data[0];
    for (int i = 1; i < n; i++) {
        if (data[i] < *min_val) *min_val = data[i];
        if (data[i] > *max_val) *max_val = data[i];
    }
    for (int i = 0; i < n; i++) {
        data[i] = (data[i] - *min_val) / (*max_val - *min_val);
    }
}

int main() {
    double x[MAX_DATA_POINTS], y[MAX_DATA_POINTS];
    int n = 0;
    FILE *file = fopen("data.csv", "r");
    if (file == NULL) {
        printf("Error: Unable to open file.\n");
        return 1;
    }

    // Load data from CSV
    while (fscanf(file, "%lf,%lf", &x[n], &y[n]) == 2) {
        n++;
        if (n >= MAX_DATA_POINTS) break;
    }
    fclose(file);

    if (n == 0) {
        printf("Error: Dataset is empty.\n");
        return 1;
    }

    // Normalize data
    double min_x, max_x, min_y, max_y;
    normalize(x, n, &min_x, &max_x);
    normalize(y, n, &min_y, &max_y);

    // Gradient Descent
    double m = 0.0, b = 0.0;
    double learning_rate = 0.01;
    int iterations = 1000;

    for (int i = 0; i < iterations; i++) {
        double dm = 0.0, db = 0.0;
        for (int j = 0; j < n; j++) {
            double prediction = m * x[j] + b;
            double error = prediction - y[j];
            dm += (2.0 / n) * error * x[j];
            db += (2.0 / n) * error;
        }

        if (isnan(dm) || isnan(db)) {
            printf("Numerical instability detected at iteration %d. dm: %f, db: %f\n", i, dm, db);
            break;
        }

        m -= learning_rate * dm;
        b -= learning_rate * db;
    }

    // Denormalize coefficients
    m = m * (max_y - min_y) / (max_x - min_x);
    b = b * (max_y - min_y) + min_y;

    printf("Trained Model: y = %fx + %f\n", m, b);

    // Predict
    double input;
    printf("Enter a value to predict: ");
    if (scanf("%lf", &input) != 1) {
        printf("Invalid input.\n");
        return 1;
    }

    double normalized_input = (input - min_x) / (max_x - min_x);
    double prediction = m * normalized_input + b;
    printf("Predicted value: %f\n", prediction);

    return 0;
}

