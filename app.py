import streamlit as st
import pandas as pd
import numpy as np
from itertools import permutations

class TSPSolver:
    def __init__(self):
        self.cities = []
        self.start_city = None
        self.cost_matrix = None

    def add_city(self, city):
        if city in self.cities:
            raise ValueError(f"City '{city}' already exists. Please enter a different city name.")
        self.cities.append(city)

    def set_start_city(self, city):
            if city in self.cities:
                self.start_city = city
            else:
                raise ValueError(f"City '{city}' not found in the list of added cities. Please add the city first.")

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

            # Consider the user-selected starting city
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

# Streamlit app
def main():
    st.title("Traveling Salesman Problem Solver")

    attempts = st.session_state.get("attempts", [])

    selected_attempt = st.selectbox("Select attempt", [f"Attempt {i+1}" for i in range(len(attempts))] + ["New Attempt"])

    # Delete button
    if st.button("Delete Attempt") and selected_attempt != "New Attempt":
        attempts = [attempt for i, attempt in enumerate(attempts) if i != int(selected_attempt.split()[-1]) - 1]
        st.session_state.attempts = attempts
        selected_attempt = "New Attempt"  # Reset to a new attempt after deletion

    if selected_attempt == "New Attempt":
        tsp_solver = TSPSolver()
        attempts.append(tsp_solver)
        st.session_state.attempts = attempts
    else:
        tsp_solver = attempts[int(selected_attempt.split()[-1]) - 1]

    # Add City section
    st.subheader("Add City")
    city = st.text_input("Enter place name:")
    if st.button("Add City"):
        try:
            tsp_solver.add_city(city)
            st.success(f"City '{city}' added successfully!")
        except ValueError as e:
            st.error(str(e))

    # Set Matrix Costs section
    st.subheader("Set Matrix Costs")
    if tsp_solver.cities:
        tsp_solver.cost_matrix = create_matrix_table(tsp_solver.cities)

      # Create an empty placeholder for the matrix
        matrix_placeholder = st.empty()
        
        # Allow user to input costs in the matrix
        for i in range(len(tsp_solver.cities)):
            for j in range(i + 1, len(tsp_solver.cities)):
                cost = st.number_input(f"Enter cost between {tsp_solver.cities[i]} and {tsp_solver.cities[j]}:")
                tsp_solver.set_cost(tsp_solver.cities[i], tsp_solver.cities[j], cost)
                
                # Dynamically update the matrix table
                matrix_placeholder.table(tsp_solver.cost_matrix)

        # Set the start city
        start_city = st.selectbox("Select start place:", tsp_solver.cities)
        tsp_solver.set_start_city(start_city)

        if st.button("Solve TSP"):
            try:
                result, cost = tsp_solver.solve_tsp()
                route = ' -> '.join(result)
                st.subheader("Optimal Path:")
                st.write(route)
                st.subheader("Total Cost:")
                st.write(cost)            

            except ValueError as e:
                st.error(str(e))

if __name__ == "__main__":
    main()
