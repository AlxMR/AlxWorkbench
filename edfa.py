from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sympy import symbols, Function, Eq, Derivative, dsolve, classify_ode, latex, simplify
from sympy.parsing.sympy_parser import parse_expr
from pydantic import BaseModel

app = FastAPI()

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ed-frontend-theta.vercel.app"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
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
    "Ecuación exacta": r"M(x, y) + N(x, y)\frac{dy}{dx} = 0 \Rightarrow \frac{\partial M}{\partial y} = \frac{\partial N}{\partial x}",
    "No se pudo determinar un método específico.": r"\text{No hay fórmula general disponible}",
}

# Función para determinar si la ecuación es ordinaria
def is_ordinary(eq):
    return eq.has(Derivative) and all(arg == x for arg in eq.free_symbols)

# Función para determinar si la ecuación es homogénea
def is_homogeneous(eq):
    try:
        from sympy import homogeneous_order
        return homogeneous_order(eq.rhs, x, y) is not None
    except:
        return False

# Función para determinar si la ecuación es lineal de primer orden
def is_linear_first_order(eq):
    try:
        dsolve(eq, y, hint='1st_linear')
        return True
    except:
        return False

@app.post("/solve-ode")
async def solve_ode(request: EquationRequest):
    equation_input = request.equation

    try:
        # Convertir la entrada en una expresión simbólica
        local_dict = {'y': y, 'x': x, 'Derivative': Derivative}
        equation = parse_expr(equation_input, local_dict=local_dict)

        if isinstance(equation, Eq):
            eq = equation
        else:
            eq = Eq(equation, 0)

        classification = classify_ode(eq, y)
        equation_type = 'Ordinaria' if is_ordinary(eq) else 'Parcial'
        equation_order = 1 if '1st' in str(classification) else None
        is_linear = 'linear' in str(classification)
        is_homog = is_homogeneous(eq)

        if not classification:
            method = "No se pudo determinar un método específico."
        else:
            best_classification = classification[0]
            if is_linear_first_order(eq):
                method = "Ecuaciones lineales de primer orden"
            elif best_classification == "Bernoulli":
                method = "Ecuación de Bernoulli"
            elif best_classification == "separable":
                method = "Separación de variables"
            elif best_classification == "1st_exact":
                method = "Ecuación exacta"
            else:
                method = "No se pudo determinar un método específico."

        try:
            if method == "Ecuación exacta":
                solution = dsolve(eq, y, hint='1st_exact')
            else:
                solution = dsolve(eq, y)
            solution_latex = latex(solution)
        except NotImplementedError:
            solution_latex = "No se pudo encontrar una solución analítica."

        recommended_formula = METHOD_FORMULAS.get(method, r"\text{No hay fórmula general disponible}")

        return {
            "classification": {
                "type": equation_type,
                "order": equation_order,
                "linearity": 'Lineal' if is_linear else 'No lineal',
                "homogeneity": 'Homogénea' if is_homog else 'No homogénea',
            },
            "method": method,
            "recommended_formula": recommended_formula,
            "solution": solution_latex,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar la ecuación: {e}")

@app.options("/")
async def handle_options():
    return {"message": "OK"}

@app.options("/solve-ode")
async def handle_solve_ode_options():
    return {"message": "OK"}
