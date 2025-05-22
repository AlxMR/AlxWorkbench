from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sympy import (symbols, Function, Eq, Derivative, dsolve, classify_ode, 
                   latex, simplify, solve, exp, sin, cos, laplace_transform)
from sympy.parsing.sympy_parser import parse_expr
from sympy.abc import s
from pydantic import BaseModel

app = FastAPI()

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todos los orígenes temporalmente para pruebas
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo para la solicitud
class EquationRequest(BaseModel):
    equation: str

# Definir símbolos
x = symbols('x', real=True)
y_func = Function('y')  # Definimos la función y, pero no la evaluamos aún

METHOD_FORMULAS = {
    "Separación de variables": r"\frac{dy}{dx} = g(x)h(y) \Rightarrow \int \frac{1}{h(y)} \, dy = \int g(x) \, dx",
    "Ecuaciones lineales de primer orden": r"\frac{dy}{dx} + P(x)y = Q(x)",
    "Ecuación de Bernoulli": r"\frac{dy}{dx} + P(x)y = Q(x)y^n",
    "Ecuación exacta": r"M(x, y) + N(x, y)\frac{dy}{dx} = 0 \Rightarrow \frac{\partial M}{\partial y} = \frac{\partial N}{\partial x}",
    "No se pudo determinar un método específico.": r"\text{No hay fórmula general disponible}",
}

@app.post("/laplace-transform")
async def laplace_transform_endpoint(request: EquationRequest):
    try:
        # Creamos y(x) para esta solicitud específica
        y = y_func(x)
        
        local_dict = {
            'y': y,
            'Y': y,
            'x': x,
            's': s,
            'Derivative': Derivative,
            'diff': Derivative,
            'exp': exp,
            'sin': sin,
            'cos': cos,
            'Function': Function
        }

        # Parsear la ecuación
        expr = parse_expr(request.equation, local_dict=local_dict)

        # Aplicar transformada de Laplace
        laplace_expr = laplace_transform(expr, x, s, noconds=True)
        simplified = simplify(laplace_expr)

        return {
            "original_equation": latex(expr),
            "laplace_transform": latex(simplified),
            "simplified": latex(simplified)
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error al calcular la transformada: {str(e)}"
        )
