import re
import patsy

def get_terms(formula, indiv_suffix='__i'):
    """extracts outcome, individual-level, group-level, and random effect
    terms from a formula.

    Arguments:

        formula: str. Formula in lmer format.

        indiv_suffix: str (default: "__i"). String representing the suffix that
            is attached to the end of individual-level terms.

    Returns:

        tuple: (outcome, indiv_terms, gp_terms, randeff_terms).

            outcome: str. Outcome variable.

            indiv_terms: list of str. Individual-level predictors.

            gp_terms: list of str. Group-level predictors.

            randeff_terms: list of str. Random effect terms.
    """
    termlist = patsy.ModelDesc.from_formula(formula).rhs_termlist
    term_names = [term.name() for term in termlist]
    indiv_terms = [re.sub(r'{0}$'.format(indiv_suffix), '', name) for name in term_names if name.endswith(indiv_suffix)]
    gp_terms = [name for name in term_names if not name.endswith(indiv_suffix) and not '|' in name]
    randeff_terms = [name for name in term_names if '|' in name]
    outcome = formula.split('~')[0].strip()
    return outcome, indiv_terms, gp_terms, randeff_terms
