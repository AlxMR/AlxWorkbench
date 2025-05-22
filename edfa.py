from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sympy import (symbols, Function, Eq, Derivative, dsolve, classify_ode, 
                   latex, simplify, solve, exp, sin, cos, laplace_transform)
from sympy.parsing.sympy_parser import parse_expr
from sympy.abc import s, t
from pydantic import BaseModel

app = FastAPI()

# Configuración CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo de solicitud
class EquationRequest(BaseModel):
    equation: str

# Símbolos globales
x = symbols('x', real=True)
y = Function('y')  # Función no evaluada

METHOD_FORMULAS = {
    "Separable": r"\frac{dy}{dx} = g(x)h(y) \Rightarrow \int \frac{1}{h(y)} dy = \int g(x) dx",
    "First-order linear": r"\frac{dy}{dx} + P(x)y = Q(x)",
    "Bernoulli": r"\frac{dy}{dx} + P(x)y = Q(x)y^n",
    "Exact": r"M(x,y)dx + N(x,y)dy = 0 \text{ con } \frac{\partial M}{\partial y} = \frac{\partial N}{\partial x}",
    "Unknown": r"\text{Método no identificado}"
}

def is_ordinary(eq):
    return eq.has(Derivative) and all(arg == x for arg in eq.free_symbols)

def is_homogeneous(eq):
    try:
        y_x = y(x)
        dy_dx = Derivative(y_x, x)
        lhs, rhs = (eq.lhs, eq.rhs) if isinstance(eq, Eq) else (eq, 0)
        expr = lhs - rhs
        dy_dx_expr = solve(expr, dy_dx)[0]
        t = symbols('t')
        return dy_dx_expr.subs(y_x, t*x).simplify() == dy_dx_expr.subs(y_x/x, t).simplify()
    except:
        return False

def is_linear_first_order(eq):
    try:
        y_x = y(x)
        dsolve(eq, y_x, hint='1st_linear')
        return True
    except:
        return False

def is_bernoulli(eq):
    try:
        y_x = y(x)
        dsolve(eq, y_x, hint='Bernoulli')
        return True
    except:
        return False

def is_exact(eq):
    try:
        y_x = y(x)
        dsolve(eq, y_x, hint='1st_exact')
        return True
    except:
        return False

@app.post("/solve-ode")
async def solve_ode(request: EquationRequest):
    try:
        y_x = y(x)  # Versión evaluada
        local_dict = {
            'y': y_x,
            'x': x,
            'Derivative': Derivative,
            'exp': exp,
            'sin': sin,
            'cos': cos
        }
        
        equation = parse_expr(request.equation, local_dict=local_dict)
        eq = equation if isinstance(equation, Eq) else Eq(equation, 0)

        classification = classify_ode(eq, y_x)
        equation_type = 'Ordinary' if is_ordinary(eq) else 'Partial'
        equation_order = max([int(hint[:1]) for hint in classification if hint[:1].isdigit()], default=None)
        is_linear = 'linear' in str(classification)
        is_homog = is_homogeneous(eq)

        if not classification:
            method = "Unknown"
        else:
            if 'separable' in classification:
                method = "Separable"
            elif is_linear_first_order(eq):
                method = "First-order linear"
            elif is_bernoulli(eq):
                method = "Bernoulli"
            elif is_exact(eq):
                method = "Exact"
            else:
                method = "Unknown"

        try:
            if method == "Exact":
                solution = dsolve(eq, y_x, hint='1st_exact')
            elif method == "Bernoulli":
                solution = dsolve(eq, y_x, hint='Bernoulli')
            elif method == "First-order linear":
                solution = dsolve(eq, y_x, hint='1st_linear')
            elif method == "Separable":
                solution = dsolve(eq, y_x, hint='separable')
            else:
                solution = dsolve(eq, y_x)
            solution_latex = latex(solution)
        except NotImplementedError:
            solution_latex = "No se encontró solución analítica."

        return {
            "classification": {
                "type": equation_type,
                "order": equation_order,
                "linearity": 'Linear' if is_linear else 'Nonlinear',
                "homogeneity": 'Homogeneous' if is_homog else 'Non-homogeneous',
            },
            "method": method,
            "formula": METHOD_FORMULAS.get(method),
            "solution": solution_latex,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")

@app.post("/laplace-transform")
async def laplace_transform_endpoint(request: EquationRequest):
    try:
        y_x = y(x)  # Versión evaluada
        local_dict = {
            'y': y_x,
            'x': x,
            's': s,
            'Derivative': Derivative,
            'exp': exp,
            'sin': sin,
            'cos': cos
        }

        expr = parse_expr(request.equation, local_dict=local_dict)
        laplace_expr = laplace_transform(expr, x, s, noconds=True)
        simplified = simplify(laplace_expr)

        return {
            "original": latex(expr),
            "transform": latex(simplified),
            "simplified": latex(simplified)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en Laplace: {str(e)}")

@app.get("/")
async def root():
    return {"message": "API de ecuaciones diferenciales"}

@app.options("/{path:path}")
async def options_handler(path: str):
    return {"message": "OK"}
