import math

R = 8.3145  # J/mol/K

# Pure component PR parameters
def calcab_pr(Tc, Pc_bar, omega, T):
    Pc = Pc_bar * 1e5  # bar -> Pa

    kappa = 0.37464 + 1.54226*omega - 0.26992*omega*omega
    alpha = (1.0 + kappa*(1.0 - math.sqrt(T/Tc)))**2

    a = 0.45724 * R*R * Tc*Tc * alpha / Pc
    b = 0.07780 * R * Tc / Pc

    Xi = -kappa*math.sqrt(T/Tc) / (1.0 + kappa*(1.0 - math.sqrt(T/Tc)))

    return a, b, Xi


# Mixing rules
def calcabmix(a11, b11, a22, b22, x1, k12=0.0):
    x2 = 1.0 - x1
    a12 = math.sqrt(a11*a22) * (1.0 - k12)

    a = x1*x1*a11 + 2.0*x1*x2*a12 + x2*x2*a22
    b = x1*b11 + x2*b22

    return a, b, a12


# PR cubic coefficients
def calccoeffs_pr(A, B):
    C2 = -1.0 + B
    C1 = A - 3.0*B*B - 2.0*B
    C0 = -A*B + B*B + B*B*B
    return C0, C1, C2



# Solve cubic
def solvecubic(C0, C1, C2):
    Q1 = C2*C1/6.0 - C0/2.0 - (C2**3)/27.0
    P1 = (C2**2)/9.0 - C1/3.0
    D = Q1*Q1 - P1*P1*P1

    if D >= 0:
        Temp1 = abs(Q1 + math.sqrt(D))**(1/3)
        Temp1 *= (Q1 + math.sqrt(D)) / abs(Q1 + math.sqrt(D))

        Temp2 = abs(Q1 - math.sqrt(D))**(1/3)
        Temp2 *= (Q1 - math.sqrt(D)) / abs(Q1 - math.sqrt(D))

        Z0 = Temp1 + Temp2 - C2/3.0
        Z1 = None
        Z2 = None
    else:
        Temp1 = Q1*Q1/(P1*P1*P1)
        Temp2 = math.sqrt(1.0 - Temp1)/math.sqrt(Temp1)
        Temp2 *= Q1/abs(Q1)

        Phi = math.atan(Temp2)
        if Phi < 0:
            Phi += math.pi

        Z0 = 2.0*math.sqrt(P1)*math.cos(Phi/3.0) - C2/3.0
        Z1 = 2.0*math.sqrt(P1)*math.cos((Phi + 2.0*math.pi)/3.0) - C2/3.0
        Z2 = 2.0*math.sqrt(P1)*math.cos((Phi + 4.0*math.pi)/3.0) - C2/3.0

        # sort descending like JS
        roots = [Z0, Z1, Z2]
        roots.sort(reverse=True)
        Z0, Z1, Z2 = roots

    return Z0, Z1, Z2, D


# Fugacity coefficient (PR)
def calcfugacity(Z, A, B, x1, Bi, A1i, A2i):
    if Z is None:
        return None

    # Guard against invalid log
    if Z - B <= 0:
        return None
    if Z - 0.41421356*B <= 0:
        return None

    temp1 = -math.log(Z - B) + Bi*(Z - 1.0)/B
    temp3 = math.log((Z + 2.41421356*B)/(Z - 0.41421356*B))

    x2 = 1.0 - x1
    temp2 = A*(2.0*(x1*A1i + x2*A2i)/A - Bi/B)/(2.82842713*B)

    return math.exp(temp1 - temp2*temp3)


# Cubic EOS wrapper
def cubic(T, P_bar, x1, Tc1, Pc1, W1, Tc2, Pc2, W2, k12=0.0):
    a11, b11, Xi1 = calcab_pr(Tc1, Pc1, W1, T)
    a22, b22, Xi2 = calcab_pr(Tc2, Pc2, W2, T)

    a, b, a12 = calcabmix(a11, b11, a22, b22, x1, k12=k12)

    P = P_bar * 1e5

    A = a * P / (R*R*T*T)
    A11 = a11 * P / (R*R*T*T)
    A22 = a22 * P / (R*R*T*T)
    A12 = a12 * P / (R*R*T*T)

    B = b * P / (R*T)
    B11 = b11 * P / (R*T)
    B22 = b22 * P / (R*T)

    C0, C1, C2 = calccoeffs_pr(A, B)
    Z0, Z1, Z2, D = solvecubic(C0, C1, C2)

    # Fugacity coefficients
    if Z2 is not None:
        phi1 = calcfugacity(Z0, A, B, x1, B11, A11, A12)
        phi2 = calcfugacity(Z0, A, B, x1, B22, A12, A22)

        phisecond1 = calcfugacity(Z2, A, B, x1, B11, A11, A12)
        phisecond2 = calcfugacity(Z2, A, B, x1, B22, A12, A22)
    else:
        phi1 = calcfugacity(Z0, A, B, x1, B11, A11, A12)
        phi2 = calcfugacity(Z0, A, B, x1, B22, A12, A22)

        Z2 = 0.0
        phisecond1 = 0.0
        phisecond2 = 0.0

    return Z0, phi1, phi2, Z2, phisecond1, phisecond2


# Initial K estimate
def Kestimate(T, P_bar, Tc, Pc_bar, omega):
    Temp = math.log(Pc_bar / P_bar)
    Temp1 = math.log(10.0)*(1.0 - Tc/T)*(7.0 + 7.0*omega)/3.0
    return math.exp(Temp + Temp1)


def Testimate(P_bar, Tc, Pc_bar, omega):
    Temp = math.log(P_bar / Pc_bar)
    Temp1 = 1.0 - Temp*3.0/(math.log(10.0)*(7.0 + 7.0*omega))
    return Tc/Temp1


# Bubble point temperature
def bubble_point_T(P_bar, x1, Tc1, Pc1, W1, Tc2, Pc2, W2, k12=0.0, T_guess=None):
    x2 = 1.0 - x1

    if T_guess is None or T_guess <= 0:
        temp1 = Testimate(P_bar, Tc1, Pc1, W1)
        temp2 = Testimate(P_bar, Tc2, Pc2, W2)
        T = x1*temp1 + x2*temp2
    else:
        T = float(T_guess)

    K1 = Kestimate(T, P_bar, Tc1, Pc1, W1)
    K2 = Kestimate(T, P_bar, Tc2, Pc2, W2)

    y1 = K1*x1
    y2 = K2*x2

    Tformer = 0.0
    i = 0

    while (abs(T - Tformer) > 1e-4 or abs(y1 - y1former) > 1e-6 or i < 20):
        Tformer = T
        y1former = y1
        i += 1

        # first get liquid fugacity coefficients
        Z0, phi1_liq, phi2_liq, Z2, phi1_vap, phi2_vap = cubic(
            T, P_bar, x1, Tc1, Pc1, W1, Tc2, Pc2, W2, k12=k12
        )

        # choose vapour fugacity set
        if Z2 > 0:
            phisecond1 = phi1_vap
            phisecond2 = phi2_vap
        else:
            phisecond1 = phi1_liq
            phisecond2 = phi2_liq

        # inner iteration loop
        for _ in range(25):
            temp = y1 + y2
            y1 /= temp
            y2 /= temp

            Z0, phi1, phi2, Z2, _, _ = cubic(
                T, P_bar, y1, Tc1, Pc1, W1, Tc2, Pc2, W2, k12=k12
            )

            y1 = x1 * phisecond1 / phi1
            y2 = x2 * phisecond2 / phi2

        temp = y1 + y2
        T = T + 0.1*T*(1.0 - temp)/temp

        if abs(y1 - y1former) < 1e-6 and abs(T - Tformer) < 1e-4 and i >= 20:
            break

    # normalize final vapour composition
    s = y1 + y2
    y1 /= s
    y2 /= s

    return T, y1, y2


# Dew point temperature
def dew_point_T(P_bar, y1, Tc1, Pc1, W1, Tc2, Pc2, W2, k12=0.0, T_guess=None):
    y2 = 1.0 - y1

    if T_guess is None or T_guess <= 0:
        temp1 = Testimate(P_bar, Tc1, Pc1, W1)
        temp2 = Testimate(P_bar, Tc2, Pc2, W2)
        T = y1*temp1 + y2*temp2
    else:
        T = float(T_guess)

    K1 = Kestimate(T, P_bar, Tc1, Pc1, W1)
    K2 = Kestimate(T, P_bar, Tc2, Pc2, W2)

    x1 = y1 / K1
    x2 = y2 / K2

    Tformer = 0.0
    i = 0

    while (abs(T - Tformer) > 1e-4 or abs(x1 - x1former) > 1e-6 or i < 20):
        Tformer = T
        x1former = x1
        i += 1

        # vapour fugacity coefficients at y
        Z0, phi1, phi2, _, _, _ = cubic(
            T, P_bar, y1, Tc1, Pc1, W1, Tc2, Pc2, W2, k12=k12
        )

        # inner loop for x
        for _ in range(25):
            temp = x1 + x2
            x1 /= temp
            x2 /= temp

            Z0, phi1_liq, phi2_liq, Z2, phi1_vap, phi2_vap = cubic(
                T, P_bar, x1, Tc1, Pc1, W1, Tc2, Pc2, W2, k12=k12
            )

            # choose liquid fugacity coefficients
            if Z2 > 0:
                phisecond1 = phi1_vap
                phisecond2 = phi2_vap
            else:
                phisecond1 = phi1_liq
                phisecond2 = phi2_liq

            x1 = y1 * phi1 / phisecond1
            x2 = y2 * phi2 / phisecond2

        temp = x1 + x2
        T = T + 0.1*T*(temp - 1.0)/temp

        if abs(x1 - x1former) < 1e-6 and abs(T - Tformer) < 1e-4 and i >= 20:
            break

    # normalize final liquid composition
    s = x1 + x2
    x1 /= s
    x2 /= s

    return T, x1, x2


### EXAMPLE USE ###
# Units are K for Tc and bar for Pc, compositions are mole fractions
# Results need sanity checking for validity in case of convergence failure or invalid roots
if __name__ == "__main__":
    
    # Component 1: N2
    Tc1, Pc1, W1 = 126.1, 33.94, 0.040

    # Component 2: C4F7CN 
    Tc2, Pc2, W2 = 385.928, 25.028, 0.352

    k12 = 0.0

    # Mixture
    y1 = 0.80  # N2 vapour fraction
    x1 = 0.80  # N2 liquid fraction

    # Bubble point at 1 bar
    Tbub, y1b, y2b = bubble_point_T(P_bar=1.0, x1=x1,
                                   Tc1=Tc1, Pc1=Pc1, W1=W1,
                                   Tc2=Tc2, Pc2=Pc2, W2=W2,
                                   k12=k12, T_guess=100)

    print("Bubble point T (K):", Tbub)
    print("Vapour y1, y2:", y1b, y2b)

    # Dew point at 1 bar
    Tdew, x1d, x2d = dew_point_T(P_bar=1.0, y1=y1,
                                Tc1=Tc1, Pc1=Pc1, W1=W1,
                                Tc2=Tc2, Pc2=Pc2, W2=W2,
                                k12=k12, T_guess=200)

    print("Dew point T (K):", Tdew)
    print("Liquid x1, x2:", x1d, x2d)