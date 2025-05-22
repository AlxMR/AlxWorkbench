from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sympy import (
    symbols, Function, Eq, Derivative, dsolve, classify_ode, latex, simplify, solve, 
    exp, sin, cos, laplace_transform, inverse_laplace_transform, Wild
)
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application
)
from pydantic import BaseModel

# Inicializar app
app = FastAPI()

# Configuración CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "https://ed-frontend-theta.vercel.app"],  # Puedes limitar aquí si deseas
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelos
class LaplaceRequest(BaseModel):
    equation: str
    initial_conditions: dict = None

class EquationRequest(BaseModel):
    equation: str

# Variables simbólicas
x = symbols('x')
y = Function('y')(x)
t = symbols('t', real=True, positive=True)
s = symbols('s')

transformations = (standard_transformations + (implicit_multiplication_application,))

# ================= LAPACE ENDPOINTS =================

def prepare_laplace_environment():
    return {
        't': t,
        's': s,
        'y': Function('y')(t),
        'Y': Function('Y')(s),
        'dy': Derivative(Function('y')(t), t),
        'd2y': Derivative(Function('y')(t), t, t),
        'exp': exp,
        'sin': sin,
        'cos': cos,
        'Derivative': Derivative,
        'Function': Function
    }

@app.post("/laplace-transform")
async def calculate_laplace(request: LaplaceRequest):
    try:
        env = prepare_laplace_environment()
        expr = parse_expr(request.equation, local_dict=env, transformations=transformations)

        if request.initial_conditions:
            for cond, value in request.initial_conditions.items():
                cond_expr = parse_expr(cond, local_dict=env)
                env.update({cond_expr: value})

        L_expr = laplace_transform(expr, t, s, noconds=True)
        simplified = simplify(L_expr)

        return {
            "time_domain": latex(expr),
            "laplace_transform": latex(simplified),
            "simplified_form": latex(simplified)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en transformada: {str(e)}")

@app.post("/inverse-laplace")
async def calculate_inverse_laplace(request: LaplaceRequest):
    try:
        env = prepare_laplace_environment()
        expr = parse_expr(request.equation, local_dict=env, transformations=transformations)
        inv_L = inverse_laplace_transform(expr, s, t)
        simplified = simplify(inv_L)

        return {
            "laplace_domain": latex(expr),
            "time_domain": latex(simplified),
            "simplified_form": latex(simplified)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en transformada inversa: {str(e)}")

# ================= ODE SOLVER ENDPOINT =================

METHOD_FORMULAS = {
    "Separación de variables": r"\frac{dy}{dx} = g(x)h(y) \Rightarrow \int \frac{1}{h(y)} \, dy = \int g(x) \, dx",
    "Ecuaciones lineales de primer orden": r"\frac{dy}{dx} + P(x)y = Q(x)",
    "Ecuación de Bernoulli": r"\frac{dy}{dx} + P(x)y = Q(x)y^n",
    "Ecuación exacta": r"M(x, y) + N(x, y)\frac{dy}{dx} = 0 \Rightarrow \frac{\partial M}{\partial y} = \frac{\partial N}{\partial x}",
    "No se pudo determinar un método específico.": r"\text{No hay fórmula general disponible}",
}

def is_ordinary(eq):
    return eq.has(Derivative) and all(arg == x for arg in eq.free_symbols)

def is_homogeneous(eq):
    try:
        dy_dx = Derivative(y, x)
        lhs, rhs = eq.lhs, eq.rhs if isinstance(eq, Eq) else (eq, 0)
        expr = lhs - rhs
        dy_dx_expr = solve(expr, dy_dx)[0]
        t_ = Wild('t')
        return dy_dx_expr.subs(y, t_ * x).simplify() == dy_dx_expr.subs(y / x, t_).simplify()
    except:
        return False

def is_linear_first_order(eq):
    try:
        dsolve(eq, y, hint='1st_linear')
        return True
    except:
        return False

def is_bernoulli(eq):
    try:
        dsolve(eq, y, hint='Bernoulli')
        return True
    except:
        return False

def is_exact(eq):
    try:
        dsolve(eq, y, hint='1st_exact')
        return True
    except:
        return False

@app.post("/solve-ode")
async def solve_ode(request: EquationRequest):
    try:
        local_dict = {'y': y, 'x': x, 'Derivative': Derivative}
        eq_input = parse_expr(request.equation, local_dict=local_dict)
        eq = eq_input if isinstance(eq_input, Eq) else Eq(eq_input, 0)

        classification = classify_ode(eq, y)
        equation_type = 'Ordinaria' if is_ordinary(eq) else 'Parcial'
        equation_order = 1 if '1st' in str(classification) else None
        is_linear = 'linear' in str(classification)
        is_homog = is_homogeneous(eq)

        if not classification:
            method = "No se pudo determinar un método específico."
        elif 'separable' in classification:
            method = "Separación de variables"
        elif is_linear_first_order(eq):
            method = "Ecuaciones lineales de primer orden"
        elif is_bernoulli(eq):
            method = "Ecuación de Bernoulli"
        elif is_exact(eq):
            method = "Ecuación exacta"
        else:
            method = "No se pudo determinar un método específico."

        try:
            hint_map = {
                "Ecuación exacta": '1st_exact',
                "Ecuación de Bernoulli": 'Bernoulli',
                "Ecuaciones lineales de primer orden": '1st_linear',
                "Separación de variables": 'separable',
            }
            hint = hint_map.get(method)
            solution = dsolve(eq, y, hint=hint) if hint else dsolve(eq, y)
            solution_latex = latex(solution)
        except NotImplementedError:
            solution_latex = "No se pudo encontrar una solución analítica."

        return {
            "classification": {
                "type": equation_type,
                "order": equation_order,
                "linearity": 'Lineal' if is_linear else 'No lineal',
                "homogeneity": 'Homogénea' if is_homog else 'No homogénea',
            },
            "method": method,
            "recommended_formula": METHOD_FORMULAS.get(method),
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

