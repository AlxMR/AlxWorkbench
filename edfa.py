from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sympy import symbols, Function, Eq, Derivative, dsolve, classify_ode, latex
from sympy.parsing.sympy_parser import parse_expr
from pydantic import BaseModel

app = FastAPI()

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todos los orígenes (ajusta según sea necesario)
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Modelo para la solicitud
class EquationRequest(BaseModel):
    equation: str

# Definir símbolos y funciones
x = symbols('x')
y = Function('y')(x)

# Fórmulas generales de los métodos recomendados
METHOD_FORMULAS = {
    "Separación de variables": r"\frac{dy}{dx} = g(x)h(y) \Rightarrow \int \frac{1}{h(y)} \, dy = \int g(x) \, dx",
    "Ecuaciones lineales de primer orden": r"\frac{dy}{dx} + P(x)y = Q(x)",
    "Ecuación de Bernoulli": r"\frac{dy}{dx} + P(x)y = Q(x)y^n",
    "No se pudo determinar un método específico.": r"\text{No hay fórmula general disponible}",
}

def classify_equation(eq):
    """
    Clasifica la ecuación diferencial de manera más precisa.
    """
    # Verificar si la ecuación es lineal de primer orden
    if eq.is_linear() and eq.is_Ordinary and eq.is_FirstOrder:
        return "Ecuaciones lineales de primer orden"
    
    # Verificar si la ecuación es separable
    if eq.is_Separable:
        return "Separación de variables"
    
    # Verificar si la ecuación es de Bernoulli
    if eq.is_Bernoulli:
        return "Ecuación de Bernoulli"
    
    # Si no se puede clasificar, devolver un mensaje genérico
    return "No se pudo determinar un método específico."

@app.post("/solve-ode")
async def solve_ode(request: EquationRequest):
    equation_input = request.equation

    try:
        # Convertir la entrada en una expresión simbólica
        equation = parse_expr(equation_input, local_dict={'y': y, 'x': x, 'Derivative': Derivative})
        eq = Eq(equation, 0)

        # Clasificación de la ecuación
        classification = classify_ode(eq, y)
        method = classify_equation(eq)  # Usar la nueva función de clasificación

        # Solución de la ecuación
        try:
            solution = dsolve(eq, y)
            solution_latex = latex(solution)
        except NotImplementedError:
            solution_latex = "No se pudo encontrar una solución analítica."

        # Obtener la fórmula general del método recomendado
        recommended_formula = METHOD_FORMULAS.get(method, r"\text{No hay fórmula general disponible}")

        # Respuesta
        return {
            "classification": {
                "type": 'Ordinaria' if 'ordinary' in str(classification) else 'Parcial',
                "order": classification[1] if len(classification) >= 2 else None,
                "linearity": 'Lineal' if len(classification) >= 3 and classification[2] else 'No lineal',
                "homogeneity": 'Homogénea' if len(classification) >= 4 and classification[3] else 'No homogénea',
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
