from fastapi import FastAPI
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI(title="Credit Union Loan Rate API")


class LoanBase(BaseModel):
    city: str
    state: str
    credit_score: int
    credit_union_name: str


class HomeLoanRequest(LoanBase):
    loan_value: float
    amount_paid: float


def get_gemini_client():
    client = genai.Client()
    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    config = types.GenerateContentConfig(tools=[grounding_tool])
    return client, config


def generate_prompt(loan_type: str, request: BaseModel) -> str:
    if loan_type == "home":
        return f"""
        The user lives in {request.city}, {request.state}, has a credit score of {request.credit_score},
        and a home loan valued at ${request.loan_value:,.2f} with ${request.amount_paid:,.2f} paid off.
        Get the current home loan interest rates from {request.credit_union_name}
        for 10, 15, and 30 terms.

        Return the results in the following JSON format:
        {{
          "{request.credit_union_name}": {{
            "10": {{"interest": "x.xx", "monthly_payment": "xxxx"}},
            "15": {{"interest": "x.xx", "monthly_payment": "xxxx"}},
            "30": {{"interest": "x.xx", "monthly_payment": "xxxx"}}
          }}
        }}
        """

    elif loan_type == "personal":
        return f"""
        The user lives in {request.city}, {request.state}, has a credit score of {request.credit_score}.
        Get the current personal loan interest rates from {request.credit_union_name}
        for 1, 3, and 5 year terms.

        Return the results in the following JSON format:
        {{
          "{request.credit_union_name}": {{
            "1": {{"interest": "x.xx", "monthly_payment": "xxxx"}},
            "3": {{"interest": "x.xx", "monthly_payment": "xxxx"}},
            "5": {{"interest": "x.xx", "monthly_payment": "xxxx"}}
          }}
        }}
        """

    elif loan_type == "credit_card":
        return f"""
        The user lives in {request.city}, {request.state}, has a credit score of {request.credit_score}.
        Get the current credit card interest rates (APR only) from {request.credit_union_name}.

        Return the results in the following JSON format:
        {{
          "{request.credit_union_name}": {{
            "low_rate_card": {{"APR": "x.xx"}},
            "cash_back_card": {{"APR": "x.xx"}},
            "rewards_card": {{"APR": "x.xx"}}
          }}
        }}
        """


# ----------- Endpoints -----------


@app.post("/get_home_loan_rates")
def get_home_loan_rates(request: HomeLoanRequest):
    client, config = get_gemini_client()
    prompt = generate_prompt("home", request)
    response = client.models.generate_content(
        model="gemini-2.5-pro", contents=prompt, config=config
    )
    return {"response": response.text.strip()}


@app.post("/get_personal_loan_rates")
def get_personal_loan_rates(request: LoanBase):
    client, config = get_gemini_client()
    prompt = generate_prompt("personal", request)
    response = client.models.generate_content(
        model="gemini-2.5-pro", contents=prompt, config=config
    )
    return {"response": response.text.strip()}


@app.post("/get_credit_card_rates")
def get_credit_card_rates(request: LoanBase):
    client, config = get_gemini_client()
    prompt = generate_prompt("credit_card", request)
    response = client.models.generate_content(
        model="gemini-2.5-pro", contents=prompt, config=config
    )
    return {"response": response.text.strip()}


@app.get("/")
def root():
    return {"message": "Welcome to the Credit Union Loan Rate API!"}
