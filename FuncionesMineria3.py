import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# ===============================================================
#   1. Varianza explicada
# ===============================================================

def plot_varianza_explicada(var_explicada, n_components):
    """
    Representa la variabilidad explicada por cada componente principal.

    Args:
        var_explicada (array): Porcentaje de varianza explicada por cada CP.
        n_components (int): Número total de componentes principales.
    """  
    num_componentes_range = np.arange(1, n_components + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(num_componentes_range, var_explicada, marker='o')

    plt.xlabel('Número de Componentes Principales')
    plt.ylabel('Varianza Explicada')
    plt.title('Variabilidad Explicada por Componente Principal')
    plt.xticks(num_componentes_range)
    plt.grid(True)

    plt.bar(num_componentes_range, var_explicada, width=0.2, align='center', alpha=0.7)
    plt.show()



# ===============================================================
#   2. Heatmap de cos²
# ===============================================================

def plot_cos2_heatmap(cosenos2):
    """
    Genera un mapa de calor de los cuadrados de las cargas (cos²).

    Args:
        cosenos2 (pd.DataFrame): Filas=variables, columnas=componentes principales.
    """
    plt.figure(figsize=(8, 8))
    sns.heatmap(cosenos2, cmap='Blues', linewidths=0.5, annot=False)

    plt.xlabel('Componentes Principales')
    plt.ylabel('Variables')
    plt.title('Cuadrados de las Cargas en las Componentes Principales')
    plt.show()



# ===============================================================
#   3. Círculos de corr + cos²  (FUNCIÓN QUE FALLABA)
# ===============================================================

def plot_corr_cos_2(n_components, correlaciones_datos_con_cp):
    """
    Representa los vectores de correlación de cada variable sobre pares de componentes.
    El color de los vectores indica la suma de cos² de cada variable.
    """

    cmap = plt.get_cmap('coolwarm')

    for i in range(n_components):
        for j in range(i + 1, n_components):

            sum_cos2 = (
                correlaciones_datos_con_cp.iloc[:, i] ** 2 +
                correlaciones_datos_con_cp.iloc[:, j] ** 2
            )

            fig, ax = plt.subplots(figsize=(10, 10))

            # Círculo unidad
            circle = plt.Circle((0, 0), 1, fill=False, color='b', linestyle='dotted')
            ax.add_patch(circle)

            # Dibujar vectores
            for k, var_name in enumerate(correlaciones_datos_con_cp.index):

                x = correlaciones_datos_con_cp.iloc[k, i]
                y = correlaciones_datos_con_cp.iloc[k, j]

                color = cmap(sum_cos2.iloc[k])

                ax.quiver(0, 0, x, y, angles='xy', scale_units='xy', scale=1, color=color)
                ax.text(x, y, var_name, color=color, fontsize=12,
                        ha='right', va='bottom')

            # Ejes
            ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
            ax.axvline(0, color='black', linestyle='--', linewidth=0.8)

            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)

            ax.set_xlabel(f'Componente Principal {i + 1}')
            ax.set_ylabel(f'Componente Principal {j + 1}')

            # Colorbar asociada a ax (solución definitiva)
            norm = plt.Normalize(vmin=sum_cos2.min(), vmax=sum_cos2.max())
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            plt.colorbar(sm, ax=ax, orientation='vertical', label='cos²')

            ax.grid()
            plt.show()



# ===============================================================
#   4. Barras de cos² por variable
# ===============================================================

def plot_cos2_bars(cos2):
    """
    Gráfico de barras de la suma de cos² por variable.

    Args:
        cos2 (pd.DataFrame): Cos² por variable y componente.
    """
    plt.figure(figsize=(8, 6))
    sns.barplot(x=cos2.sum(axis=1), y=cos2.index, color="blue")

    plt.xlabel('Suma de los $cos^2$')
    plt.ylabel('Variables')
    plt.title('Varianza explicada por cada variable (cos²)')
    plt.show()

# ===============================================================
#   5. Contribuciones proporcionales
# ===============================================================

def plot_contribuciones_proporcionales(cos2, autovalores, n_components):
    """
    Cálculo y visualización de contribuciones proporcionales.

    Args:
        cos2 (DataFrame)
        autovalores (array)
        n_components (int)
    """
    contribuciones = cos2 * np.sqrt(autovalores)

    sumas_contribuciones = [
        np.sum(contribuciones[f'Componente {i + 1}'])
        for i in range(n_components)
    ]

    contribuciones_prop = contribuciones.div(sumas_contribuciones, axis=1) * 100

    plt.figure(figsize=(8, 8))
    sns.heatmap(contribuciones_prop, cmap='Blues', linewidths=0.5, annot=False)

    plt.xlabel('Componentes Principales')
    plt.ylabel('Variables')
    plt.title('Contribuciones Proporcionales (%)')
    plt.show()

    return contribuciones_prop



# ===============================================================
#   6. Scatter de PCA
# ===============================================================

def plot_pca_scatter(pca, datos_estandarizados, n_components):
    """
    Gráficos de observaciones en pares de componentes principales.
    """
    componentes = pca.transform(datos_estandarizados)

    for i in range(n_components):
        for j in range(i + 1, n_components):

            plt.figure(figsize=(8, 6))
            plt.scatter(componentes[:, i], componentes[:, j])

            for k, etiqueta in enumerate(datos_estandarizados.index):
                plt.annotate(etiqueta, (componentes[k, i], componentes[k, j]))

            plt.axhline(0, color='black', linestyle='--')
            plt.axvline(0, color='black', linestyle='--')

            plt.xlabel(f'Componente {i + 1}')
            plt.ylabel(f'Componente {j + 1}')
            plt.title('PCA – Observaciones')
            plt.show()



# ===============================================================
#   7. Scatter con vectores de variables
# ===============================================================

def plot_pca_scatter_with_vectors(pca, datos_estandarizados, n_components, components_):
    """
    Scatter PCA con vectores de contribución de variables.
    """
    componentes = pca.transform(datos_estandarizados)

    for i in range(n_components):
        for j in range(i + 1, n_components):

            plt.figure(figsize=(8, 6))
            plt.scatter(componentes[:, i], componentes[:, j])

            for k, etiqueta in enumerate(datos_estandarizados.index):
                plt.annotate(etiqueta, (componentes[k, i], componentes[k, j]))

            plt.axhline(0, color='black', linestyle='--')
            plt.axvline(0, color='black', linestyle='--')

            plt.xlabel(f'Componente {i + 1}')
            plt.ylabel(f'Componente {j + 1}')
            plt.title('PCA observaciones + vectores')

            fit = pca.fit(datos_estandarizados)
            coeff = fit.components_.T
            scaled = 8 * coeff

            for v in range(scaled.shape[0]):
                plt.arrow(0, 0, scaled[v, i], scaled[v, j], color='red', alpha=0.5)
                plt.text(scaled[v, i], scaled[v, j],
                         datos_estandarizados.columns[v],
                         color='red')

            plt.show()



# ===============================================================
#   8. Scatter con categorías
# ===============================================================

def plot_pca_scatter_with_categories(datos_componentes_sup_var, componentes_principales_sup, n_components, var_categ):
    """
    PCA con categorías y centroides.
    """
    categorias = datos_componentes_sup_var[var_categ].unique()

    for i in range(n_components):
        for j in range(i + 1, n_components):

            plt.figure(figsize=(8, 6))
            plt.scatter(componentes_principales_sup[:, i], componentes_principales_sup[:, j])

            for categoria in categorias:
                obs = componentes_principales_sup[datos_componentes_sup_var[var_categ] == categoria]
                centro = np.mean(obs, axis=0)
                plt.scatter(centro[i], centro[j], label=categoria, s=100)

            for k, etiqueta in enumerate(datos_componentes_sup_var.index):
                plt.annotate(etiqueta, (componentes_principales_sup[k, i],
                                        componentes_principales_sup[k, j]))

            plt.axhline(0, color='black', linestyle='--')
            plt.axvline(0, color='black', linestyle='--')

            plt.xlabel(f'Componente {i + 1}')
            plt.ylabel(f'Componente {j + 1}')
            plt.title('PCA por categorías')
            plt.legend()
            plt.show()
