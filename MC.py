import numpy as np
import matplotlib.pyplot as plt

def calculate_jacobian(t):
    N = len(t)
    jacobian = np.zeros((N, 2))
    for i in range(N):
        jacobian[i, 0] = -t[i]
        jacobian[i, 1] = -1
    return jacobian
# Fonction pour calculer les valeurs optimales de a et b
def linear_least_squares(t, v):
    J = calculate_jacobian(t)
    a, b = -np.linalg.inv(J.T @ J) @ J.T @ v
    return a, b
t = np.array([2, 4, 5])
v = np.array([2, 4.3, 4.9])
# Calcul a et b
a, b = linear_least_squares(t, v)
# Points pour la droite calculée
t_line = np.linspace(min(t), max(t), 100)
v_line = a * t_line + b
# Tracer les points d'observation et la droite calculée
plt.scatter(t, v, label="Points d'observation")
plt.plot(t_line, v_line, label="Droite calculée", color='red')
plt.xlabel("t")
plt.ylabel("v")
plt.legend()
plt.title("Approximation linéaire des points d'observation")
plt.show()
print(f"Paramètres de la droite calculée : a = {a}, b = {b}")
# Générer des données simulées avec une droite y = 2x + 1 et ajout de bruit aléatoire
np.random.seed(0)  # Pour reproduire les mêmes résultats
N = 100  # Nombre de points
x_simulated = np.linspace(0, 10, N)
y_simulated = 2 * x_simulated + 1 + np.random.normal(0, 1, N)
# Appliquer la méthode d'approximation linéaire aux données simulées
a_optimal, b_optimal = linear_least_squares(x_simulated, y_simulated)
# Tracer les points d'observations et la droite calculée
plt.scatter(x_simulated, y_simulated, label="Données simulées")
plt.plot(x_simulated, a_optimal * x_simulated + b_optimal,
         color='red', label="Droite approximative")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Approximation linéaire des données simulées")
plt.show()
print(f"Valeurs optimales de a et b : a = {a_optimal}, b = {b_optimal}")
