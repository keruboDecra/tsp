import streamlit as st
import numpy as np
import pandas as pd
from itertools import permutations

class TSPSolver:
    def __init__(self):
        self.cities = []
        self.start_city = None
        self.cost_matrix = None

    def add_city(self, city):
        self.cities.append(city)

    def set_start_city(self, city):
        self.start_city = city

    def set_cost(self, city1, city2, cost):
        index1 = self.cities.index(city1)
        index2 = self.cities.index(city2)
        self.cost_matrix.at[city1, city2] = cost
        self.cost_matrix.at[city2, city1] = cost

    def solve_tsp(self):
        # Generate all possible permutations of cities
        all_permutations = permutations(self.cities)

        # Calculate the total cost for each permutation
        total_costs = []
        for perm in all_permutations:
            cost = 0
            for i in range(len(perm) - 1):
                cost += self.cost_matrix.at[perm[i], perm[i + 1]]
            cost += self.cost_matrix.at[perm[-1], perm[0]]  # Return to the starting city

            # Consider the starting city
            if self.start_city:
                cost += self.cost_matrix.at[self.start_city, perm[0]]

            total_costs.append((perm + (perm[0],), cost))  # Append the starting city to the permutation

        # Find the permutation with the minimum total cost
        min_permutation, min_cost = min(total_costs, key=lambda x: x[1])

        return min_permutation, min_cost

def create_matrix_table(cities):
    size = len(cities)
    matrix = pd.DataFrame(np.zeros((size, size), dtype=float), index=cities, columns=cities)
    return matrix

def render_matrix_table(matrix):
    st.dataframe(matrix.style.highlight_max(axis=0, color='lightgreen').highlight_max(axis=1, color='lightgreen'))

def input_matrix_values(matrix, cities):
    for i in range(len(cities)):
        for j in range(i + 1, len(cities)):
            cost = st.number_input(f"Enter cost between {cities[i]} and {cities[j]}:", min_value=0.0)
            matrix.at[cities[i], cities[j]] = cost
            matrix.at[cities[j], cities[i]] = cost
    return matrix

def main():
    st.title("TSP Solver")

    tsp_solver = TSPSolver()

    while True:
        st.sidebar.header("Menu")
        option = st.sidebar.selectbox("Choose an option:", ["Add City", "Set Start City", "Set Cost Matrix", "Solve TSP", "Exit"])

        if option == "Add City":
            city = st.text_input("Enter city name:")
            if city:
                tsp_solver.add_city(city)
        elif option == "Set Start City":
            start_city = st.selectbox("Select start city:", tsp_solver.cities, index=0)
            tsp_solver.set_start_city(start_city)
        elif option == "Set Cost Matrix":
            if tsp_solver.cities:
                tsp_solver.cost_matrix = create_matrix_table(tsp_solver.cities)
                render_matrix_table(tsp_solver.cost_matrix)
                tsp_solver.cost_matrix = input_matrix_values(tsp_solver.cost_matrix, tsp_solver.cities)
                render_matrix_table(tsp_solver.cost_matrix)
            else:
                st.warning("Please add cities first.")
        elif option == "Solve TSP":
            if tsp_solver.cost_matrix is not None:
                result, cost = tsp_solver.solve_tsp()
                route = ' -> '.join(result)
                st.success(f"Optimal Path: {route}")
                st.success(f"Total Cost: {cost}")
            else:
                st.warning("Please set the cost matrix first.")
        elif option == "Exit":
            break

if __name__ == "__main__":
    main()
