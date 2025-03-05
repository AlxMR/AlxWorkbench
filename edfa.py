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

        # Determinar si la ecuación es ordinaria o parcial
        is_ordinary = 'ordinary' in str(classification)
        equation_type = 'Ordinaria' if is_ordinary else 'Parcial'

        # Determinar el orden de la ecuación
        equation_order = 1 if '1st' in str(classification) else None

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
                "type": equation_type,  # Ordinaria o Parcial
                "order": equation_order,  # Orden de la ecuación
                "linearity": 'Lineal' if best_classification == "1st_linear" else 'No lineal',
                "homogeneity": 'Homogénea' if "homogeneous" in classification else 'No homogénea',
            },
            "method": method,
            "recommended_formula": recommended_formula,  # Fórmula general del método recomendado
            "solution": solution_latex,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar la ecuación: {e}")
