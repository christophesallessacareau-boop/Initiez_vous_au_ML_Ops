Home Credit default risk:
Prédire la probabilité de faillite d'un client.

6 notebooks sont réalisés correspondant chacun à l'analyse exploratoire d'un fichier data :
1_bureau_bureau_balance.ipynb
2_credit_card_balance.ipynb
3_installments_payments.ipynb
4_previous_application.ipynb
5_pos_cash_balance.ipynb
6_application_train.ipynb

Déroulement:
1_Bureau_bureau_balance.ipynb est crée à partir de 2 sources (Bureau et bureau_balance, fusionnés via la clé SK_ID_BUREAU-Identifiant d’un crédit externe).
1_Bureau_bureau_balance.ipynb sera fusionné avec 6_application_train via la clé principale SD_ID_CURR (identifiant unique de chaque demande de prêt)

2_credit_card_balance.ipynb sera fusionné avec 6_application_train.ipynb via la clé SD_ID_CURR

3_installments_payment.ipynb a été fusionné avec 6_application_train.ipynb suite à la clé commune SK_ID_PREV (Identifiant d’un ancien crédit) avec 4_previous_application.ipynb

4_previous_application.ipynb sera fusionné avec 6_application_train.ipynb via la clé SD_ID_CURR

5_pos_cash_balance.ipynb sera fusionné avec 6_application_train.ipynb via la clé SD_ID_CURR

6_application_train.ipynb est le principal fichier:
-détient la clé SD_ID_CURR permettant de fusionner les autres fichiers avec ce fichier
-comprend la variable cible que l'on étudie (client solvable / client en faillite)

Une analyse exploratoire et pré-processing sont réalisés sur chacun de ces 6 fichiers.
Leur fusion est effective dans le notebook main.ipynb

main.ipynb:
-fusions des 6 fichiers entre eux
-entrainement de différents modèles
-recherche du meilleur score (le meilleur AUC et le moindre coût) pour l'estimation de la variable cible.
-option: données SHAP (visualisation des critères expliquant la variable cible: globalement ou par individu)

NB: AUC du gagnant du concours Kaggle Home Credit default risk = 0.82