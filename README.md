# Codecademy
# Yelp Restaurant Rating Prediction – Multiple Lineær Regresjon

##  Prosjektbeskrivelse  
Dette prosjektet bruker ekte Yelp-data for å undersøke hvilke faktorer som påvirker en restaurants rating.  
Målet er å trene en **multiple linear regression-modell** for å predikere en restaurants gjennomsnittlige stjernerating basert på ulike egenskaper.  

## Datasett  
Prosjektet bruker flere JSON-filer fra Yelp:  
- **yelp_business.json** – informasjon om restauranter (adresse, kategori, fasiliteter, m.m.)  
- **yelp_review.json** – oppsummerte anmeldelser og sentimentanalyse  
- **yelp_user.json**, **yelp_checkin.json**, **yelp_tip.json** (ikke brukt i endelig modell)  

## Brukte verktøy  
- **Python**  
- **Jupyter Notebook**  
- **Pandas / NumPy** – databehandling  
- **Seaborn / Matplotlib** – visualisering  
- **scikit-learn** – modelltrening og evaluering  

## Prosess  
1. **Datainnhenting** – Leste inn og slo sammen business- og review-data på `business_id`.  
2. **Rensing** – Fjernet manglende verdier og ubrukte kolonner.  
3. **Feature Selection** – Beholdt features med korrelasjon ≥ 0.05 mot target (`stars`).  
4. **Skalering** – Standardiserte data med `StandardScaler`.  
5. **Modelltrening** – Trente en Multiple Linear Regression-modell.  
6. **Evaluering** – Målte ytelse med R² og RMSE.  
7. **Forbedring** – Testet polynomial features (degree=2) for å øke ytelsen.  

## Resultater  
| Modell                     | R²     | RMSE   |
|----------------------------|--------|--------|
| Linear Regression          | 0.6786 | 0.5738 |
| Linear Regression (scaled) | 0.6691 | 0.5821 |
| Polynomial Features (deg=2)| 0.7127 | 0.5425 |

## Viktige funn  
- Størst påvirkning på rating hadde: antall anmeldelser, gjennomsnittlig sentiment-score, prisnivå, og visse fasiliteter (f.eks. Wi-Fi, reservasjoner).  
- Skalering alene ga ikke bedre resultat i denne modellen.  
- Polynomial features ga en merkbar forbedring i både R² og RMSE.  

## Videre arbeid  
- Teste regularisering (Ridge / Lasso) for å redusere overtilpasning.  
- Prøve andre modeller (Random Forest, Gradient Boosting) for sammenligning.  
- Utforske tekstdata fra anmeldelser for mer detaljert sentimentanalyse.  

## Oppsummering
Prosjektet bygde en multiple linear regression-modell for å forutsi Yelp-rating basert på restaurantdata.
Etter dataforberedelse og feature selection ble nøyaktigheten forbedret:

R²: 0.678 → 0.713

RMSE: 0.574 → 0.543

Resultatet viser at modellen kan forutsi ratinger relativt godt, men at det fortsatt finnes rom for forbedring.
