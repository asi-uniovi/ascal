"""
AWS instance classes for family c5/m5/r5 and region eu-west-1 (Ireland)
Important. This file is an example, so prices and instances do not have to agree with the real ones.
"""

from cloudmodel.unified.units import ComputationalUnits, CurrencyPerTime, Storage
from fcma import InstanceClass, InstanceClassFamily

# Instance class families. Firstly, the parent family and next its children
c5_m5_r5_fm = InstanceClassFamily("c5_m5_r5")
c5_fm = InstanceClassFamily("c5", parent_fms=c5_m5_r5_fm)
m5_fm = InstanceClassFamily("m5", parent_fms=c5_m5_r5_fm)
r5_fm = InstanceClassFamily("r5", parent_fms=c5_m5_r5_fm)

families = [
    c5_m5_r5_fm,
    c5_fm,
    m5_fm,
    r5_fm,
]

# Instance classes
c5_large = InstanceClass(
    name="c5.large",
    price=CurrencyPerTime("0.096 usd/hour"),
    cores=ComputationalUnits("1 cores"),
    mem=Storage("4 gibibytes"),
    family=c5_fm,
)
c5_xlarge = c5_large.mul(2, "c5.xlarge")
c5_2xlarge = c5_xlarge.mul(2, "c5.2xlarge")
c5_4xlarge = c5_xlarge.mul(4, "c5.4xlarge")
c5_9xlarge = c5_xlarge.mul(9, "c5.9xlarge")
c5_12xlarge = c5_xlarge.mul(12, "c5.12xlarge")
c5_18xlarge = c5_xlarge.mul(18, "c5.18xlarge")
c5_24xlarge = c5_xlarge.mul(24, "c5.24xlarge")

m5_large = InstanceClass(
    name="m5.large",
    price=CurrencyPerTime("0.107 usd/hour"),
    cores=ComputationalUnits("1 cores"),
    mem=Storage("8 gibibytes"),
    family=m5_fm,
)
m5_xlarge = m5_large.mul(2, "m5.xlarge")
m5_2xlarge = m5_xlarge.mul(2, "m5.2xlarge")
m5_4xlarge = m5_xlarge.mul(4, "m5.4xlarge")
m5_9xlarge = m5_xlarge.mul(9, "m5.9xlarge")
m5_12xlarge = m5_xlarge.mul(12, "m5.12xlarge")
m5_18xlarge = m5_xlarge.mul(18, "m5.18xlarge")
m5_24xlarge = m5_xlarge.mul(24, "m5.24xlarge")

r5_large = InstanceClass(
    name="r5.large",
    price=CurrencyPerTime("0.141 usd/hour"),
    cores=ComputationalUnits("1 cores"),
    mem=Storage("16 gibibytes"),
    family=r5_fm,
)
r5_xlarge = r5_large.mul(2, "r5.xlarge")
r5_2xlarge = r5_xlarge.mul(2, "r5.2xlarge")
r5_4xlarge = r5_xlarge.mul(4, "r5.4xlarge")
r5_9xlarge = r5_xlarge.mul(9, "r5.9xlarge")
r5_12xlarge = r5_xlarge.mul(12, "r5.12xlarge")
r5_18xlarge = r5_xlarge.mul(18, "r5.18xlarge")
r5_24xlarge = r5_xlarge.mul(24, "r5.24xlarge")

