"""Policy (STAND-IN): placeholder citations until retrieval over policy/sources/ exists.

The real agent will retrieve the Selling Guide, Reg B and HMDA passages this
profile raises (RAG) and return them as citations.
"""
from __future__ import annotations

from mrn_shared import Citation, PolicyResult

from ._common import current, dump

SG = "Fannie Mae Selling Guide"


def policy(state):
    scen = current(state)
    p, d = scen.profile.purchase, scen.derived
    cites = [Citation(source=SG, section="(stand-in)",
                      excerpt="Placeholder: the Policy agent will cite the eligibility passages for this purchase.")]
    if d.ltv_pct > 80:
        cites.append(Citation(source=SG, section="(stand-in)",
                              excerpt="Placeholder: down payments under 20% normally mean mortgage insurance; "
                                      "the Policy agent will cite the exact passage."))
    if p.property_type.value in ("condo", "townhouse"):
        cites.append(Citation(source=SG, section="(stand-in)",
                              excerpt="Placeholder: condo (and some townhouse) purchases involve a project "
                                      "eligibility review; the Policy agent will cite the exact passage."))
    return {"policy": dump(PolicyResult(source="stub", citations=cites))}
