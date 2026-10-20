import os
import pandas as pd
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------
load_dotenv()


api_key = os.getenv("GROQ_API_KEY")
model_name = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)

if not api_key:
    raise ValueError(
        "GROQ_API_KEY is missing. Add it to your .env file."
    )

if not model_name:
    raise ValueError(
        "GROQ_MODEL is missing. Set a supported model ID in .env."
    )

# --------------------------------------------------
# 2. Load vehicle dataset
# --------------------------------------------------
df = pd.read_csv("carbon_data.csv")

required_columns = [
    "MODELYEAR", "MAKE", "MODEL", "VEHICLECLASS",
    "ENGINESIZE", "CYLINDERS", "TRANSMISSION",
    "FUELTYPE", "FUELCONSUMPTION_CITY",
    "FUELCONSUMPTION_HWY", "FUELCONSUMPTION_COMB",
    "FUELCONSUMPTION_COMB_MPG", "CO2EMISSIONS"
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing dataset columns: {missing}")

df = df.dropna(subset=required_columns).reset_index(drop=True)

if df.empty:
    raise ValueError("No complete vehicle records are available.")

# --------------------------------------------------
# 3. Vehicle selection tool
# --------------------------------------------------
def select_vehicle(dataframe, index=0):
    """Select a vehicle and return its details."""
    if index < 0 or index >= len(dataframe):
        raise IndexError("Vehicle index is outside the dataset.")

    vehicle = dataframe.iloc[index]

    return {
        "year": vehicle["MODELYEAR"],
        "make": vehicle["MAKE"],
        "model": vehicle["MODEL"],
        "vehicle_class": vehicle["VEHICLECLASS"],
        "engine_size": vehicle["ENGINESIZE"],
        "cylinders": vehicle["CYLINDERS"],
        "transmission": vehicle["TRANSMISSION"],
        "fuel_type": vehicle["FUELTYPE"],
        "city": vehicle["FUELCONSUMPTION_CITY"],
        "highway": vehicle["FUELCONSUMPTION_HWY"],
        "combined": vehicle["FUELCONSUMPTION_COMB"],
        "mpg": vehicle["FUELCONSUMPTION_COMB_MPG"],
        "co2": vehicle["CO2EMISSIONS"]
    }


vehicle = select_vehicle(df, index=0)

print("Selected vehicle:")
print(pd.Series(vehicle).to_string())

# --------------------------------------------------
# 4. Initialize Groq LLM
# --------------------------------------------------
llm = ChatGroq(
    model=model_name,
    temperature=0.2,
    api_key=api_key
)

parser = StrOutputParser()

# --------------------------------------------------
# 5. Agent 1: Fuel Efficiency Analyst
# --------------------------------------------------
efficiency_prompt = ChatPromptTemplate.from_template("""
You are a Fuel Efficiency Analysis Agent.

Analyze this vehicle:
Make: {make}
Model: {model}
Year: {year}
Vehicle class: {vehicle_class}
Engine size: {engine_size} L
Fuel type: {fuel_type}
City consumption: {city} L/100 km
Highway consumption: {highway} L/100 km
Combined consumption: {combined} L/100 km
Combined fuel economy: {mpg} MPG

Explain:
1. Fuel efficiency.
2. City versus highway consumption.
3. Likely fuel-saving opportunities.
4. Limitations of the available data.

Do not invent benchmark values. Explain that lower L/100 km
means better fuel efficiency.
""")

efficiency_agent = efficiency_prompt | llm | parser

# --------------------------------------------------
# 6. Agent 2: Carbon Emission Analyst
# --------------------------------------------------
emission_prompt = ChatPromptTemplate.from_template("""
You are a Vehicle Carbon Emission Analysis Agent.

Vehicle: {year} {make} {model}
Vehicle class: {vehicle_class}
Fuel type: {fuel_type}
Combined fuel consumption: {combined} L/100 km
CO2 emissions: {co2} g/km

Analyze:
1. The meaning of the reported CO2 value.
2. Factors that may influence vehicle emissions.
3. Environmental implications.
4. Data limitations.

Do not claim that the reported value represents total
lifecycle emissions. It is a vehicle emissions measure.
Do not invent regulatory thresholds or comparisons.
""")

emission_agent = emission_prompt | llm | parser

# --------------------------------------------------
# 7. Tool: Transparent Eco-Friendliness Score
# --------------------------------------------------
def calculate_eco_score(co2, combined_consumption):
    """
    Illustrative rule-based score, not an official rating.

    The score uses configurable illustrative reference points:
    0-100 g/km CO2 and 0-15 L/100 km combined consumption.
    """
    co2 = max(float(co2), 0.0)
    consumption = max(float(combined_consumption), 0.0)

    co2_score = max(0, min(10, 10 * (1 - co2 / 300)))
    fuel_score = max(
        0, min(10, 10 * (1 - consumption / 20))
    )

    score = round(0.7 * co2_score + 0.3 * fuel_score, 1)

    if score >= 7:
        category = "Relatively lower impact"
    elif score >= 4:
        category = "Moderate impact"
    else:
        category = "Higher impact"

    return {
        "score": score,
        "category": category,
        "method": (
            "Illustrative weighted score: 70% CO2 component "
            "and 30% fuel-consumption component. "
            "Not an official environmental rating."
        )
    }


eco_result = calculate_eco_score(
    vehicle["co2"],
    vehicle["combined"]
)

# --------------------------------------------------
# 8. Agent 3: Sustainability Recommendation Agent
# --------------------------------------------------
recommendation_prompt = ChatPromptTemplate.from_template("""
You are an Automotive Sustainability Recommendation Agent.

Vehicle: {year} {make} {model}
Fuel type: {fuel_type}
Combined consumption: {combined} L/100 km
CO2 emissions: {co2} g/km
Illustrative eco score: {score}/10
Score category: {category}

Provide:
1. Three practical recommendations to reduce emissions.
2. Explain how each recommendation helps.
3. Identify which actions the driver can take immediately.
4. Suggest longer-term vehicle or transport alternatives.

Do not promise exact emission reductions without supporting data.
Distinguish driving improvements from changing vehicle technology.
""")

recommendation_agent = recommendation_prompt | llm | parser

# --------------------------------------------------
# 9. Agent 4: Final Report Orchestrator
# --------------------------------------------------
report_prompt = ChatPromptTemplate.from_template("""
You are the Lead Automotive Sustainability Consultant.

Create a professional final report using the supplied
vehicle information and the three agent analyses.

VEHICLE DETAILS:
{vehicle_details}

FUEL EFFICIENCY AGENT:
{efficiency_report}

EMISSION AGENT:
{emission_report}

ECO SCORE TOOL:
{eco_score}

RECOMMENDATION AGENT:
{recommendations}

Produce these sections:
1. Executive Summary
2. Overall Environmental Impact
3. Fuel Efficiency Analysis
4. CO2 Emission Analysis
5. Environment Friendly? Yes / No / Depends, with reasons
6. Three Emission Reduction Recommendations
7. Eco-Friendliness Score out of 10
8. Final Conclusion

Use clear headings, bullet points and the actual supplied data.
Explain that the score is illustrative, not an official rating.
Do not introduce unsupported measurements or claims.
""")

report_agent = report_prompt | llm | parser

# --------------------------------------------------
# 10. Run the agent workflow
# --------------------------------------------------
print("\nRunning Fuel Efficiency Agent...")
efficiency_report = efficiency_agent.invoke(vehicle)

print("\nRunning Emission Analysis Agent...")
emission_report = emission_agent.invoke(vehicle)

print("\nRunning Recommendation Agent...")
recommendations = recommendation_agent.invoke({
    **vehicle,
    **eco_result
})

vehicle_details = "\n".join(
    f"{key}: {value}" for key, value in vehicle.items()
)

print("\nGenerating final sustainability report...")
final_report = report_agent.invoke({
    "vehicle_details": vehicle_details,
    "efficiency_report": efficiency_report,
    "emission_report": emission_report,
    "eco_score": str(eco_result),
    "recommendations": recommendations
})

# --------------------------------------------------
# 11. Display and save report
# --------------------------------------------------
print("\n" + "=" * 60)
print("VEHICLE SUSTAINABILITY REPORT")
print("=" * 60)
print(final_report)

with open(
    "vehicle_sustainability_report.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(final_report)

print("\nReport saved to vehicle_sustainability_report.txt")