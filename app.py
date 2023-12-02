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

# Streamlit app
def main():
    st.title("Traveling Salesman Problem Solver")

    # Sidebar to switch between cases
    case_index = st.sidebar.selectbox("Select a case", range(1, 6), key="case_index")

    # Initialize or retrieve TSPSolver instance for the selected case
    tsp_solver = st.session_state.get(f"tsp_solver_{case_index}", TSPSolver())

    # Main content
    st.sidebar.header("Options")
    option = st.sidebar.selectbox("Select an option", ["Add City", "Set Start City", "Set Cost Matrix"])

    if option == "Add City":
        city = st.text_input("Enter city name:")
        if st.button("Add City"):
            tsp_solver.add_city(city)
            st.success(f"City '{city}' added successfully!")

        # Allow user to edit their input of cities
        if st.button("Edit Cities"):
            st.text("Current Cities: " + ", ".join(tsp_solver.cities))
            new_cities = st.text_input("Enter new cities (comma-separated):")
            tsp_solver.cities = [city.strip() for city in new_cities.split(",") if city.strip()]
            st.success("Cities updated successfully!")

    elif option == "Set Cost Matrix":
        if tsp_solver.cities:
            tsp_solver.cost_matrix = create_matrix_table(tsp_solver.cities)
            st.table(tsp_solver.cost_matrix)

            # Allow user to input costs in the matrix
            for i in range(len(tsp_solver.cities)):
                for j in range(i + 1, len(tsp_solver.cities)):
                    cost = st.number_input(f"Enter cost between {tsp_solver.cities[i]} and {tsp_solver.cities[j]}:")
                    tsp_solver.set_cost(tsp_solver.cities[i], tsp_solver.cities[j], cost)

            # Button to set start city
            if st.button("Set Start City"):
                tsp_solver.start_city = None
                if tsp_solver.cities:
                    start_city = st.selectbox("Select start city:", tsp_solver.cities)
                    try:
                        tsp_solver.set_start_city(start_city)
                        st.success(f"Start city set to '{start_city}' successfully!")
                    except ValueError as e:
                        st.error(str(e))
                else:
                    st.warning("Please add cities first.")

            # Button to solve TSP
            if st.button("Solve TSP"):
                try:
                    result, cost = tsp_solver.solve_tsp()
                    route = ' -> '.join(result)
                    st.subheader("Optimal Path:")
                    st.write(route)
                    st.subheader("Total Cost:")
                    st.write(cost)

                    # Option to calculate legs
                    calculate_legs = st.checkbox("Calculate Legs")
                    if calculate_legs:
                        legs = len(result) - 1
                        st.subheader("Number of Legs:")
                        st.write(legs)

                except ValueError as e:
                    st.error(str(e))

    # Display added cities
    if tsp_solver.cities:
        st.sidebar.subheader("Added Cities:")
        st.sidebar.write(", ".join(tsp_solver.cities))

    # Save the current TSPSolver instance to session state
    st.session_state[f"tsp_solver_{case_index}"] = tsp_solver

if __name__ == "__main__":
    main()
