# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 2022

@author: tdrumond & agademer

Template file for your Exercise 3 submission 
(GA solving Mastermind example)
"""
from ga_solver import GAProblem
import mastermind as mm


class MastermindProblem(GAProblem):
    """Implementation of GAProblem for the mastermind problem"""
    pass  # REPLACE WITH YOUR CODE


if __name__ == '__main__':

    from ga_solver import GASolver

    match = mm.MastermindMatch(secret_size=6)
    problem = MastermindProblem(match)
    solver = GASolver(problem)

    solver.reset_population()
    solver.evolve_until()

    print(
        f"Best guess {mm.decode_guess(solver.getBestDNA())} {solver.get_best_individual()}")
    print(
        f"Problem solved? {match.is_correct(solver.get_best_individual().chromosome)}")
