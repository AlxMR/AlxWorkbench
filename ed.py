import streamlit as st
from sympy import symbols, Function, Eq, Derivative, dsolve, classify_ode, latex
from sympy.parsing.sympy_parser import parse_expr

# Configuración de la página
st.set_page_config(page_title="Reconocimiento de Ecuaciones Diferenciales", layout="wide")

# Título de la aplicación
st.title("Reconocimiento y Solución de Ecuaciones Diferenciales")

# Definir símbolos y funciones
x = symbols('x')
y = Function('y')(x)

# Entrada de la ecuación
st.header("Ingresa tu ecuación diferencial")
equation_input = st.text_input(
    "Escribe tu ecuación aquí (usa 'y' para la función y 'x' para la variable independiente):",
    "Derivative(y, x, x) - Derivative(y, x) + 6*y"
)

# Procesamiento de la ecuación
try:
    # Convertir la entrada en una expresión simbólica
    equation = parse_expr(equation_input, local_dict={'y': y, 'x': x, 'Derivative': Derivative})
    eq = Eq(equation, 0)
    
    # Clasificación de la ecuación
    st.header("Clasificación de la Ecuación")
    classification = classify_ode(eq, y)
    
    # Verificar la longitud de la tupla classification
    if len(classification) >= 1:
        st.write(f"**Tipo de ecuación:** {'Ordinaria' if 'ordinary' in str(classification) else 'Parcial'}")
    if len(classification) >= 2:
        st.write(f"**Orden:** {classification[1]}")
    if len(classification) >= 3:
        st.write(f"**Linealidad:** {'Lineal' if classification[2] else 'No lineal'}")
    if len(classification) >= 4:
        st.write(f"**Homogeneidad:** {'Homogénea' if classification[3] else 'No homogénea'}")

    # Método de solución recomendado
    st.header("Método de Solución Recomendado")
    if 'separable' in str(classification):
        st.write("**Método recomendado:** Separación de variables")
        st.latex(r"\frac{dy}{dx} = g(x)h(y) \Rightarrow \int \frac{1}{h(y)} dy = \int g(x) dx")
    elif 'linear' in str(classification):
        st.write("**Método recomendado:** Ecuaciones lineales de primer orden")
        st.latex(r"\frac{dy}{dx} + P(x)y = Q(x)")
        st.latex(r"y = e^{-\int P(x) dx} \left( \int Q(x) e^{\int P(x) dx} dx + C \right)")
    elif 'Bernoulli' in str(classification):
        st.write("**Método recomendado:** Ecuación de Bernoulli")
        st.latex(r"\frac{dy}{dx} + P(x)y = Q(x)y^n")
        st.latex(r"v = y^{1-n} \Rightarrow \frac{dv}{dx} + (1-n)P(x)v = (1-n)Q(x)")
    else:
        st.write("**Método recomendado:** No se pudo determinar un método específico.")

    # Solución de la ecuación
    st.header("Solución de la Ecuación")
    try:
        solution = dsolve(eq, y)
        st.write("**Solución general:**")
        st.latex(latex(solution))
    except NotImplementedError:
        st.write("**No se pudo encontrar una solución analítica.**")

except Exception as e:
    st.error(f"Error al procesar la ecuación: {e}")

# Ejemplos y soluciones 
st.header("Ejemplos y Soluciones ")
example = st.selectbox(
    "Selecciona un ejemplo:",
    ["Derivative(y, x, x) - Derivative(y, x) + 6*y", "Derivative(y, x) + y - x", "Derivative(y, x) - y**2"]
)

if example:
    st.write(f"**Ecuación seleccionada:** {example}")
    try:
        eq_example = parse_expr(example, local_dict={'y': y, 'x': x, 'Derivative': Derivative})
        eq_example = Eq(eq_example, 0)
        solution_example = dsolve(eq_example, y)
        st.write("**Solución paso a paso:**")
        st.latex(latex(solution_example))
    except Exception as e:
        st.error(f"Error al resolver el ejemplo: {e}")