"""
HybridMath FastAPI Backend
--------------------------
AI-powered math word problem solver using T5 + SymPy

Usage:
    uvicorn backend:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sympy import N
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application
)
import time
import re

# =============================================================================
# Initialize FastAPI App
# =============================================================================

app = FastAPI(
    title="HybridMath Solver API",
    description="AI-powered math word problem solver using T5 + SymPy",
    version="1.0.1"
)

# Allow frontend connections (adjust origin for deployment)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict later for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Model Loader
# =============================================================================

class HybridMathSolver:
    def __init__(self, model_path="./hybridmath-final"):
        print("🚀 Loading HybridMath model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            self.tokenizer = T5Tokenizer.from_pretrained(model_path)
            self.model = T5ForConditionalGeneration.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"❌ Failed to load model from {model_path}: {e}")
            raise e

    def translate_to_equation(self, problem_text):
        """Generate equation text from a natural language problem"""
        input_text = f"solve: {problem_text}"
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=128,
                num_beams=5,
                early_stopping=True
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    def clean_equation(self, equation_str):
        """Normalize and clean up equations for safe evaluation"""
        if not equation_str:
            return ""

        # Basic cleanup
        equation_str = (
            equation_str.replace('×', '*')
            .replace('÷', '/')
            .replace('^', '**')
            .replace('−', '-')
            .replace('–', '-')
            .replace('=', '')
            .strip()
        )

        # ✅ Remove all words like 'books', 'apples', 'km', etc.
        equation_str = re.sub(r"[A-Za-z_]+", "", equation_str)

        # ✅ Keep only valid math characters
        equation_str = re.sub(r"[^0-9+\-*/().]", "", equation_str)

        # ✅ Fix leading zeros (e.g., 007 → 7, 00012 → 12)
        equation_str = re.sub(r'\b0+(\d+)', r'\1', equation_str)

        return equation_str

    def evaluate_equation(self, equation_str):
        """Safely evaluate a mathematical expression"""
        try:
            equation_str = self.clean_equation(equation_str)

            if not equation_str.strip():
                return None, "Empty or invalid equation after cleaning"

            # ✅ Try using SymPy for parsing
            try:
                transformations = standard_transformations + (implicit_multiplication_application,)
                expr = parse_expr(equation_str, transformations=transformations)
                result = float(N(expr))
                return result, None
            except Exception:
                # Fallback: Python eval()
                result = eval(equation_str)
                return float(result), None

        except Exception as e:
            return None, f"Evaluation error: {str(e)}"

    def solve(self, problem_text):
        """End-to-end solver for math word problems"""
        start_time = time.time()
        try:
            equation = self.translate_to_equation(problem_text)
            answer, error = self.evaluate_equation(equation)
            time_taken = time.time() - start_time
            return equation, answer, time_taken, error
        except Exception as e:
            return None, None, time.time() - start_time, str(e)


# =============================================================================
# Global Solver Instance
# =============================================================================

solver = None

@app.on_event("startup")
async def startup_event():
    global solver
    solver = HybridMathSolver(model_path="./hybridmath-final")


# =============================================================================
# API Models
# =============================================================================

class MathProblem(BaseModel):
    problem: str

class SolutionResponse(BaseModel):
    equation: str
    answer: float | None
    time_taken: float
    status: str
    error: str | None = None


# =============================================================================
# API Routes
# =============================================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "HybridMath Solver API is running 🚀",
        "version": "1.0.1",
        "endpoints": ["/solve", "/batch-solve", "/docs"]
    }

@app.post("/solve", response_model=SolutionResponse)
async def solve_problem(problem: MathProblem):
    """Solve a single math word problem"""
    if solver is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not problem.problem.strip():
        raise HTTPException(status_code=400, detail="Problem text cannot be empty")

    equation, answer, time_taken, error = solver.solve(problem.problem)

    if error or answer is None:
        return SolutionResponse(
            equation=equation or "",
            answer=None,
            time_taken=time_taken,
            status="error",
            error=error or "Could not evaluate equation"
        )

    return SolutionResponse(
        equation=equation,
        answer=answer,
        time_taken=time_taken,
        status="success"
    )

@app.post("/batch-solve")
async def batch_solve(problems: list[MathProblem]):
    """Solve multiple math problems at once"""
    if solver is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = []
    for p in problems:
        equation, answer, time_taken, error = solver.solve(p.problem)
        results.append({
            "problem": p.problem,
            "equation": equation,
            "answer": answer,
            "time_taken": time_taken,
            "status": "success" if not error else "error",
            "error": error
        })

    return {"results": results, "total": len(results)}


# =============================================================================
# Run Command
# =============================================================================
# Run with: uvicorn backend:app --reload --host 0.0.0.0 --port 8000
# Docs: http://127.0.0.1:8000/docs
# =============================================================================
