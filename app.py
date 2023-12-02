import streamlit as st
import pandas as pd
import numpy as np
from itertools import permutations

class TSPSolver:
    def __init__(self, session_state):
        self.session_state = session_state
        if not hasattr(self.session_state, 'cities'):
            self.session_state.cities = []
        self.start_city = None
        self.cost_matrix = None

    def add_city(self, city):
        self.session_state.cities.append(city)

    def set_start_city(self, city):
        if city in self.session_state.cities:
            self.start_city = city
        else:
            raise ValueError(f"City '{city}' not found in the list of added cities. Please add the city first.")

    def set_cost(self, city1, city2, cost):
        index1 = self.session_state.cities.index(city1)
        index2 = self.session_state.cities.index(city2)
        self.cost_matrix.at[city1, city2] = cost
        self.cost_matrix.at[city2, city1] = cost

    def solve_tsp(self):
        # Generate all possible permutations of cities
        all_permutations = permutations(self.session_state.cities)

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
    session_state = st.session_state
    if not hasattr(session_state, 'tsp_solver'):
        session_state.tsp_solver = TSPSolver(session_state)

    st.title("Traveling Salesman Problem Solver")

    # Sidebar
    st.sidebar.header("Options")
    option = st.sidebar.selectbox("Select an option", ["Add City", "Set Start City", "Set Cost Matrix", "Solve TSP"])

    # Main content
    if option == "Add City":
        city = st.text_input("Enter city name:")
        if st.button("Add City"):
            session_state.tsp_solver.add_city(city)
            st.success(f"City '{city}' added successfully!")

    elif option == "Set Cost Matrix":
        if session_state.tsp_solver.session_state.cities:
            session_state.tsp_solver.cost_matrix = create_matrix_table(session_state.tsp_solver.session_state.cities)
            st.table(session_state.tsp_solver.cost_matrix)

            # Allow user to input costs in the matrix
            for i in range(len(session_state.tsp_solver.session_state.cities)):
                for j in range(i + 1, len(session_state.tsp_solver.session_state.cities)):
                    cost = st.number_input(f"Enter cost between {session_state.tsp_solver.session_state.cities[i]} and {session_state.tsp_solver.session_state.cities[j]}:")
                    session_state.tsp_solver.set_cost(session_state.tsp_solver.session_state.cities[i], session_state.tsp_solver.session_state.cities[j], cost)

            if st.button("Set Start City"):
                if session_state.tsp_solver.session_state.cities:
                    start_city = st.selectbox("Select start city:", session_state.tsp_solver.session_state.cities)
                    try:
                        session_state.tsp_solver.set_start_city(start_city)
                        st.success(f"Start city set to '{start_city}' successfully!")
                    except ValueError as e:
                        st.error(str(e))
                else:
                    st.warning("Please add cities first.")

            if session_state.tsp_solver.start_city:
                if st.button("Solve TSP"):
                    try:
                        result, cost = session_state.tsp_solver.solve_tsp()
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
    if session_state.tsp_solver.session_state.cities:
        st.sidebar.subheader("Added Cities:")
        st.sidebar.write(", ".join(session_state.tsp_solver.session_state.cities))

if __name__ == "__main__":
    main()
