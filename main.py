import dataclasses
from collections import namedtuple
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
import pandas as pd
import re
import numpy as np

month_map = {
    "Jan": "January",
    "Fev": "February",
    "Mar": "March",
    "Abr": "April",
    "Mai": "May",
    "Jun": "June",
    "Jul": "July",
    "Ago": "August",
    "Set": "September",
    "Out": "October",
    "Nov": "November",
    "Dez": "December",
}


# Define named tuples for each tariff
Tariff = namedtuple("Tariff", ["power", "kwh"])

# Define named tuple for energy
Consumption = namedtuple("Consumption", ["vazio", "cheias", "ponta"])


@dataclasses.dataclass
class TariffBase:
    power: float = dataclasses.field(default_factory=float)
    kwh_iva_reduction: float = dataclasses.field(default=100)

    def kwh_formula(self, v, c, p, d):
        raise NotImplementedError

    def formula(self, v, c, p, d):
        kwh_cost = self.kwh_formula(v, c, p, d)
        pw_cost = 31 * self.power * (1 - d)
        total_s_iva = kwh_cost + pw_cost
        # Taxas e Impostos
        kwh = sum([v, c, p])
        ratio = float(min(self.kwh_iva_reduction, kwh) / kwh)
        iva_6 = kwh_cost * ratio * IVA_6
        v = kwh_cost * (1 - ratio) + pw_cost
        dedg = 0.07
        iec = kwh * 0.001
        iva_23 = v * IVA_23 + (dedg + iec) * IVA_23
        taxas_e_impostos = iva_6 + iva_23 + dedg + iec

        # Contribuição Audiovisual
        cav = 2.85 * (1 + IVA_6)

        # Total
        total = total_s_iva + taxas_e_impostos + cav
        return total

    def cost(self, con: Consumption, discount: float):
        return self.formula(con.vazio, con.cheias, con.ponta, discount)


@dataclasses.dataclass
class Simple(TariffBase):
    simple_rate: float = dataclasses.field(default_factory=float)

    def kwh_formula(self, v, c, p, d=0.0):
        return self.simple_rate * (v + c + p) * (1 - d)


@dataclasses.dataclass
class BiHour(TariffBase):
    fora_vazio_rate: float = dataclasses.field(default_factory=float)
    vazio_rate: float = dataclasses.field(default_factory=float)

    def kwh_formula(self, v, c, p, d=0.0):
        v = self.vazio_rate * v
        fv = self.fora_vazio_rate * (c + p)
        return (v + fv) * (1 - d)


@dataclasses.dataclass
class TriHour(TariffBase):
    ponta_rate: float = dataclasses.field(default_factory=float)
    vazio_rate: float = dataclasses.field(default_factory=float)
    cheias_rate: float = dataclasses.field(default_factory=float)

    def kwh_formula(self, v, c, p, d=0.0):
        v = self.vazio_rate * v
        p = self.ponta_rate * p
        c = self.cheias_rate * c

        return (v + p + c) * (1 - d)


TARIFF = dict(
    simple=Simple(
        power=0.4160,
        simple_rate=0.2031,
    ),
    bi_hour=BiHour(power=0.4252, fora_vazio_rate=0.2431, vazio_rate=0.1480),
    tri_hour=TriHour(
        power=0.4117, ponta_rate=0.3939, cheias_rate=0.1700, vazio_rate=0.1397
    ),
)


IVA_6 = 0.06
IVA_13 = 0.13
IVA_23 = 0.23


def read(xlsx):
    # Read the Excel file
    df = pd.read_excel(xlsx, skiprows=0)
    # Drop rows and columns that are completely empty
    df.dropna(how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)

    # The header is actually the second row in the file
    df.columns = df.iloc[0]
    df = df[1:]

    # Reset index
    df.reset_index(drop=True, inplace=True)

    # Fill the first column with a placeholder since it's all NaN
    df.iloc[:, 0] = "Placeholder"

    # Define the namedtuple
    Record = namedtuple(
        "Record", [re.sub(r"\W|^(?=\d)", "_", col).lower() for col in df.columns]
    )
    #
    # # Convert dataframe to a list of namedtuples
    records = [Record(*row) for row in df.itertuples(index=False, name=None)]
    records = list(filter(lambda obj: obj.estado == "Aceite", records))

    consumptions = []
    for c, n in zip(records, records[1:]):
        d, m, y = c.envio_de_leitura.split(" ")
        value = list(
            curr - prev
            for curr, prev in zip(
                [c.vazio__kwh_, c.cheias__kwh_, c.ponta__kwh_],
                [n.vazio__kwh_, n.cheias__kwh_, n.ponta__kwh_],
            )
        )

        consumptions.append(
            (
                datetime.strptime(f"{d} {month_map[m]} {y}", "%d %B %Y"),
                Consumption(*value),
            )
        )
    return consumptions


def main():

    consumptions = read("95c6aa5f-fe7f-5665-59f4-bdcfecb031b310963350E.xlsx")

    cost_by_plan = dict(simple=[], bi_hour=[], tri_hour=[])

    # Calculate costs for each month and each plan
    for _, consumption in reversed(consumptions):
        for name, tariff in TARIFF.items():
            total = tariff.cost(consumption, discount=0.02)
            # print(f"{month.strftime('%B %Y')} - {name}: {total}")
            cost_by_plan[name].append(total)

    for plan, v in cost_by_plan.items():
        print(f"[{plan}]: total:{sum(v)}")
    # Plotting
    month_names = [month.strftime("%B %Y") for month, _ in reversed(consumptions)]

    plt.figure(figsize=(10, 6))
    plt.plot(month_names, cost_by_plan["simple"], marker="o", label="Simple Rate")
    plt.plot(month_names, cost_by_plan["bi_hour"], marker="o", label="Bi-Hourly Rate")
    plt.plot(month_names, cost_by_plan["tri_hour"], marker="o", label="Tri-Hourly Rate")
    plt.xlabel("Month")
    plt.ylabel("Cost (€)")
    plt.title("Monthly Energy Costs for Different Pricing Plans")
    plt.xticks(month_names)
    plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for better readability
    plt.grid(True)
    plt.legend()

    # Save the plot as PNG
    plt.show()


if __name__ == "__main__":
    main()
