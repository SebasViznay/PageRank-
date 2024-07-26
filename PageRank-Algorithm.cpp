// PageRank-Algorithm.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <cstdlib>
#include <ctime>
#include <chrono>

const double d = 0.85;  // Damping factor, represents the probability of continuing to follow links
const double epsilon = 1e-6;  // Convergence threshold, minimun value to check for convergence  

// Function to generate a random web graph as an adjacency matrix
std::vector<std::vector<int>> generateRandomGraph(int pages, double linkProbability) {
    std::vector<std::vector<int>> adjMatrix(pages, std::vector<int>(pages, 0)); // Initialize adjacency matrix

    std::srand(std::time(0));

    // Fill adjacency matrix with random links based on variable "linkProbability"
    for (int i = 0; i < pages; i+=1) {
        for (int j = 0; j < pages; j+=1) {
            if (i != j && (static_cast<double>(std::rand()) / RAND_MAX) < linkProbability) {
                adjMatrix[i][j] = 1; // Set link from page i to page j
            }
        }
    }

    return adjMatrix;
}

// Print pages results
void printResults(const std::vector<double>& results) {
    for (int i = 0; i < results.size(); i += 1) {
        std::cout << "Page " << i << ": " << results[i] << std::endl;
    }
}

int main() {
    int pages = 50000; // Number of pages in the web graph
    double linkProbability = 0.1; // Probability of a link between any two pages

    // Generate a random web graph
    std::vector<std::vector<int>> adjMatrix = generateRandomGraph(pages, linkProbability);

    int n = static_cast<int>(adjMatrix.size()); // Number of pages
    std::vector<double> results(n, 1.0 / n); // Initialize results to 1/N
    std::vector<double> newResults(n, 0.0); // Temporary vector for new results

    omp_set_num_threads(16);

    auto start = std::chrono::high_resolution_clock::now(); // Start timing

    bool converged; // Flag for convergence
    int iteration = 0; // Iteration counter
    do {
        converged = true; // Assume convergence
        iteration+=1;

#pragma omp parallel for // Parallelize the rank update computation
        for (int i = 0; i < n; i+=1) {
            newResults[i] = (1 - d) / n; // Teleportation factor for each page
            for (int j = 0; j < n; j+=1) {
                if (adjMatrix[j][i] == 1) { // Check if there's a link from page j to page i
                    int outLinks = std::count(adjMatrix[j].begin(), adjMatrix[j].end(), 1); // Count outgoing links from page j
                    if (outLinks > 0) {
                        newResults[i] += d * results[j] / outLinks; // Add distributed rank from page j
                    }
                }
            }
        }

#pragma omp parallel for reduction(&& : converged) // Parallelize convergence check
        for (int i = 0; i < n; i+=1) {
            if (std::abs(newResults[i] - results[i]) > epsilon) { // Check if the change is greater than epsilon
                converged = false; // If any page hasn't converged, set converged to false
            }
        }

        std::swap(results, newResults); // Swap results with newResults for the next iteration

    } while (!converged); // Continue until results converge

    auto end = std::chrono::high_resolution_clock::now(); // End timing
    std::chrono::duration<double> duration = end - start; // Calculate duration

    printResults(results); // Print final results
    std::cout << "Total time: " << duration.count() << " seconds" << std::endl; // Print time taken

    // Finding the top-ranked page

    double maxRank = results[0];
    int maxRankPage = 0;
#pragma omp parallel for
    for (int i = 1; i < n; i+=1) {
#pragma omp critical
        if (results[i] > maxRank) {
            maxRank = results[i];
            maxRankPage = i;
        }
    }
    std::cout << "Top ranked page: " << maxRankPage << " with " << maxRank << " (" << maxRank * 100 << "%)" << std::endl; // Print top-ranked page and its rank as a fraction

    return 0;
}