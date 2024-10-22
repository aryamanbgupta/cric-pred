import math
from scipy.stats import beta
from scipy.optimize import minimize_scalar

def get_betting_odds(match_info):
    # This function would typically interact with a betting API
    # For this example, we'll return a placeholder value
    return 1.909  # Example: -110 in American odds

def get_model_probability(match_info):
    # This function would typically run your predictive model
    # For this example, we'll return a placeholder value
    return 0.55  # Example: 55% probability

def implied_probability(odds):
    return 1 / odds

def kelly_criterion(p, odds):
    q = 1 - p
    return max(0, (p * odds - 1) / (odds - 1))

def modified_kelly_f0(p_hat, odds):
    return max(0, (p_hat * odds - 1) / (odds - 1))

def modified_kelly_f1(x, n, a, b, odds):
    p_tilde = beta.median(x + a, n - x + b)
    return max(0, (p_tilde * odds - 1) / (odds - 1))

def modified_kelly_f2(x, n, a, b, odds):
    def integrand(p):
        return max(0, (p * odds - 1) / (odds - 1)) * beta.pdf(p, x + a, n - x + b)
    return beta.expect(integrand, args=(x + a, n - x + b))

def loss_function_f3(f, p, odds, c1, c2, k):
    kelly = max(0, (p * odds - 1) / (odds - 1))
    if f > kelly:
        return (c1 + c2) * abs(f - kelly)**k
    else:
        return c2 * abs(f - kelly)**k

def modified_kelly_f3(x, n, a, b, odds, c1, c2, k):
    def expected_loss(f):
        return beta.expect(lambda p: loss_function_f3(f, p, odds, c1, c2, k), args=(x + a, n - x + b))
    result = minimize_scalar(expected_loss, bounds=(0, 1), method='bounded')
    return result.x

def fixed_percentage(percentage=0.02):
    return percentage

def unit_based(unit_size, bankroll, confidence):
    return (unit_size / bankroll) * confidence

def proportional(edge, max_bet=0.1):
    return min(edge, max_bet)

def main():
    bankroll = 1000  # Example bankroll
    match_info = "Team A vs Team B"  # Example match info

    # Historical data and prior parameters
    x = 100  # number of winning historical matches
    n = 180  # total number of historical matches
    a = 50   # Beta prior parameter
    b = 50   # Beta prior parameter

    odds = get_betting_odds(match_info)
    model_prob = get_model_probability(match_info)
    implied_prob = implied_probability(odds)

    print(f"Match: {match_info}")
    print(f"Betting odds: {odds}")
    print(f"Implied probability: {implied_prob:.4f}")
    print(f"Model probability: {model_prob:.4f}")

    edge = model_prob - implied_prob

    # Calculate betting fractions
    original_kelly = kelly_criterion(model_prob, odds)
    f0 = modified_kelly_f0(model_prob, odds)
    f1 = modified_kelly_f1(x, n, a, b, odds)
    f2 = modified_kelly_f2(x, n, a, b, odds)
    f3a = modified_kelly_f3(x, n, a, b, odds, 1, 1, 1.5)
    f3b = modified_kelly_f3(x, n, a, b, odds, 1, 2, 1.5)
    fixed_perc = fixed_percentage()
    unit_based_frac = unit_based(50, bankroll, edge)  # Assuming 50 as unit size
    prop_frac = proportional(edge)

    print("\nBetting fractions:")
    print(f"Original Kelly: {original_kelly:.4f}")
    print(f"Modified Kelly f0: {f0:.4f}")
    print(f"Modified Kelly f1: {f1:.4f}")
    print(f"Modified Kelly f2: {f2:.4f}")
    print(f"Modified Kelly f3a: {f3a:.4f}")
    print(f"Modified Kelly f3b: {f3b:.4f}")
    print(f"Fixed Percentage (2%): {fixed_perc:.4f}")
    print(f"Unit-based (confidence={edge:.4f}): {unit_based_frac:.4f}")
    print(f"Proportional: {prop_frac:.4f}")

    # Calculate bet amounts
    print("\nBet amounts (based on ${bankroll} bankroll):")
    print(f"Original Kelly: ${original_kelly * bankroll:.2f}")
    print(f"Modified Kelly f0: ${f0 * bankroll:.2f}")
    print(f"Modified Kelly f1: ${f1 * bankroll:.2f}")
    print(f"Modified Kelly f2: ${f2 * bankroll:.2f}")
    print(f"Modified Kelly f3a: ${f3a * bankroll:.2f}")
    print(f"Modified Kelly f3b: ${f3b * bankroll:.2f}")
    print(f"Fixed Percentage (2%): ${fixed_perc * bankroll:.2f}")
    print(f"Unit-based: ${unit_based_frac * bankroll:.2f}")
    print(f"Proportional: ${prop_frac * bankroll:.2f}")

if __name__ == "__main__":
    main()