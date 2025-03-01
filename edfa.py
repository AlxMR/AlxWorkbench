from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sympy import symbols, Function, Eq, Derivative, dsolve, classify_ode, latex
from sympy.parsing.sympy_parser import parse_expr
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ed-frontend-theta.vercel.app"],  # Dominio de tu frontend
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos (GET, POST, etc.)
    allow_headers=["*"],  # Permite todos los encabezados
)

# Modelo para la solicitud
class EquationRequest(BaseModel):
    equation: str

# Definir símbolos y funciones
x = symbols('x')
y = Function('y')(x)

@app.post("/solve-ode")
async def solve_ode(request: EquationRequest):
    equation_input = request.equation

    try:
        # Convertir la entrada en una expresión simbólica
        equation = parse_expr(equation_input, local_dict={'y': y, 'x': x, 'Derivative': Derivative})
        eq = Eq(equation, 0)

        # Clasificación de la ecuación
        classification = classify_ode(eq, y)

        # Método de solución recomendado
        method = ""
        if 'separable' in str(classification):
            method = "Separación de variables"
        elif 'linear' in str(classification):
            method = "Ecuaciones lineales de primer orden"
        elif 'Bernoulli' in str(classification):
            method = "Ecuación de Bernoulli"
        else:
            method = "No se pudo determinar un método específico."

        # Solución de la ecuación
        try:
            solution = dsolve(eq, y)
            solution_latex = latex(solution)
        except NotImplementedError:
            solution_latex = "No se pudo encontrar una solución analítica."

        # Respuesta
        return {
            "classification": {
                "type": 'Ordinaria' if 'ordinary' in str(classification) else 'Parcial',
                "order": classification[1] if len(classification) >= 2 else None,
                "linearity": 'Lineal' if len(classification) >= 3 and classification[2] else 'No lineal',
                "homogeneity": 'Homogénea' if len(classification) >= 4 and classification[3] else 'No homogénea',
            },
            "method": method,
            "solution": solution_latex,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar la ecuación: {e}")
