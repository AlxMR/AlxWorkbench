from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sympy import symbols, Function, Eq, Derivative, dsolve, classify_ode, latex
from sympy.parsing.sympy_parser import parse_expr
from pydantic import BaseModel

app = FastAPI()

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ed-frontend-theta.vercel.app"],  # Dominio de tu frontend
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # Métodos permitidos
    allow_headers=["*"],  # Encabezados permitidos
)

# Modelo para la solicitud
class EquationRequest(BaseModel):
    equation: str

# Definir símbolos y función desconocida
x = symbols('x')
y = Function('y')(x)

# Diccionario con fórmulas generales de los métodos recomendados
METHOD_FORMULAS = {
    "Separación de variables": r"\frac{dy}{dx} = g(x)h(y) \Rightarrow \int \frac{1}{h(y)} \, dy = \int g(x) \, dx",
    "Ecuaciones lineales de primer orden": r"\frac{dy}{dx} + P(x)y = Q(x)",
    "Ecuación de Bernoulli": r"\frac{dy}{dx} + P(x)y = Q(x)y^n",
    "No se pudo determinar un método específico.": r"\text{No hay fórmula general disponible}",
}

@app.post("/solve-ode")
async def solve_ode(request: EquationRequest):
    equation_input = request.equation

    try:
        # Convertir la entrada en una expresión simbólica
        local_dict = {'y': y, 'x': x, 'Derivative': Derivative}
        equation = parse_expr(equation_input, local_dict=local_dict)

        # Si la ecuación ya está igualada a cero, la usamos directamente
        if isinstance(equation, Eq):
            eq = equation
        else:
            eq = Eq(equation, 0)

        # Clasificación de la ecuación diferencial
        classification = classify_ode(eq, y)

        # Asegurar que tenemos al menos un método identificado
        if not classification:
            method = "No se pudo determinar un método específico."
        else:
            # Se toma solo la primera clasificación, que es la más específica
            best_classification = classification[0]

            # Priorizar métodos de forma correcta
            if best_classification == "Bernoulli":
                method = "Ecuación de Bernoulli"
            elif best_classification == "1st_linear":
                method = "Ecuaciones lineales de primer orden"
            elif best_classification == "separable":
                method = "Separación de variables"
            else:
                method = "No se pudo determinar un método específico."

        # Solución de la ecuación diferencial
        try:
            solution = dsolve(eq, y)
            solution_latex = latex(solution)
        except NotImplementedError:
            solution_latex = "No se pudo encontrar una solución analítica."

        # Obtener la fórmula general del método recomendado
        recommended_formula = METHOD_FORMULAS.get(method, r"\text{No hay fórmula general disponible}")

        # Construcción de la respuesta
        return {
            "classification": {
                "type": 'Ordinaria' if 'ordinary' in str(classification) else 'Parcial',
                "order": 1 if '1st' in str(classification) else None,  # Corregir el orden
                "linearity": 'Lineal' if best_classification == "1st_linear" else 'No lineal',
                "homogeneity": 'Homogénea' if "homogeneous" in classification else 'No homogénea',
            },
            "method": method,
            "recommended_formula": recommended_formula,  # Fórmula general del método recomendado
            "solution": solution_latex,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar la ecuación: {e}")

# Ruta para manejar solicitudes OPTIONS en la raíz
@app.options("/")
async def handle_options():
    return {"message": "OK"}

# Ruta para manejar solicitudes OPTIONS en /solve-ode
@app.options("/solve-ode")
async def handle_solve_ode_options():
    return {"message": "OK"}
