"""
AWS c7a, m7a and r7a instance classes for eu-west-1 (Ireland).
Linux On-Demand prices in USD/hour, checked 2026-10-27.
Specifications: https://docs.aws.amazon.com/ec2/latest/instancetypes/co.html
Prices: https://instances.vantage.sh/aws/ec2/c7a.large?region=eu-west-1
        https://instances.vantage.sh/aws/ec2/m7a.large?region=eu-west-1
        https://instances.vantage.sh/aws/ec2/r7a.large?region=eu-west-1
"""

from cloudmodel.unified.units import ComputationalUnits, CurrencyPerTime, Storage
from fcma import InstanceClass, InstanceClassFamily

# Instance class families. Firstly, the parent family and next its children
c7a_m7a_r7a_fm = InstanceClassFamily("c7a_m7a_r7a")
c7a_fm = InstanceClassFamily("c7a", parent_fms=c7a_m7a_r7a_fm)
m7a_fm = InstanceClassFamily("m7a", parent_fms=c7a_m7a_r7a_fm)
r7a_fm = InstanceClassFamily("r7a", parent_fms=c7a_m7a_r7a_fm)

families = [
    c7a_m7a_r7a_fm,
    c7a_fm,
    m7a_fm,
    r7a_fm,
]

# Instance classes (one core is two vCPUs)
c7a_large = InstanceClass(
    name="c7a.large",
    price=CurrencyPerTime("0.10264 usd/hour"),
    cores=ComputationalUnits("1 cores"),
    mem=Storage("4 gibibytes"),
    family=c7a_fm,
)
c7a_xlarge = c7a_large.mul(2, "c7a.xlarge")
c7a_2xlarge = c7a_large.mul(4, "c7a.2xlarge")
c7a_4xlarge = c7a_large.mul(8, "c7a.4xlarge")
c7a_8xlarge = c7a_large.mul(16, "c7a.8xlarge")
c7a_12xlarge = c7a_large.mul(24, "c7a.12xlarge")
c7a_16xlarge = c7a_large.mul(32, "c7a.16xlarge")
c7a_24xlarge = c7a_large.mul(48, "c7a.24xlarge")
c7a_32xlarge = c7a_large.mul(64, "c7a.32xlarge")
c7a_48xlarge = c7a_large.mul(96, "c7a.48xlarge")

m7a_large = InstanceClass(
    name="m7a.large",
    price=CurrencyPerTime("0.11592 usd/hour"),
    cores=ComputationalUnits("1 cores"),
    mem=Storage("8 gibibytes"),
    family=m7a_fm,
)
m7a_xlarge = m7a_large.mul(2, "m7a.xlarge")
m7a_2xlarge = m7a_large.mul(4, "m7a.2xlarge")
m7a_4xlarge = m7a_large.mul(8, "m7a.4xlarge")
m7a_8xlarge = m7a_large.mul(16, "m7a.8xlarge")
m7a_12xlarge = m7a_large.mul(24, "m7a.12xlarge")
m7a_16xlarge = m7a_large.mul(32, "m7a.16xlarge")
m7a_24xlarge = m7a_large.mul(48, "m7a.24xlarge")
m7a_32xlarge = m7a_large.mul(64, "m7a.32xlarge")
m7a_48xlarge = m7a_large.mul(96, "m7a.48xlarge")

r7a_large = InstanceClass(
    name="r7a.large",
    price=CurrencyPerTime("0.15215 usd/hour"),
    cores=ComputationalUnits("1 cores"),
    mem=Storage("16 gibibytes"),
    family=r7a_fm,
)
r7a_xlarge = r7a_large.mul(2, "r7a.xlarge")
r7a_2xlarge = r7a_large.mul(4, "r7a.2xlarge")
r7a_4xlarge = r7a_large.mul(8, "r7a.4xlarge")
r7a_8xlarge = r7a_large.mul(16, "r7a.8xlarge")
r7a_12xlarge = r7a_large.mul(24, "r7a.12xlarge")
r7a_16xlarge = r7a_large.mul(32, "r7a.16xlarge")
r7a_24xlarge = r7a_large.mul(48, "r7a.24xlarge")
r7a_32xlarge = r7a_large.mul(64, "r7a.32xlarge")
r7a_48xlarge = r7a_large.mul(96, "r7a.48xlarge")
