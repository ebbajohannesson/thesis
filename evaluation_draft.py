# behandla data as needed

# skapa y_true (i notebooken gjorde vi en mindre med bara resultaten för rerankingen men kan nog 
# koppla det till en för all data? om vi ger alla som inte kom med i rankingen score 0 borde det bli rätt, 
# våra metrics bryr sig ändå bara om dem som är med i top_k, inte om hur många man missade)

# skapa embeddings
# - skapa större ("vanliga/bas") embeddings
# - skapa mindre embeddings
## crossencoder med stora hela vägen
## retrieval med små+reranking med stora för att se hur mkt man tappar, retrieval med stora+reranking med bauta för att se hur mkt man tjänar
## prova GPU med stora för jämförelse?
## sen ska vi prova med mindre för en modell - hela vägen eller bara på rerankingen? det blir ju att se om man förlorar så mkt på det
 # (om man gör små+stora och stora+bauta har man typ gjort det?)

# kör retrieval med olika embeddingstorlekar
# med ANN/similarity/faiss? ska SVM göras efter det eller istället för det första urvalet?
# gör både större och mindre urval (vi skrev större och mindre dataset i måldokumentet men inte så intresserade av att testa det på vår retrieval, mer på rerankingen)

# skapa en predict/rerank-funktion för varje metod/modell (egen funktion för modell som skapar mindre/bauta embeddings - även om sequenceClassification så skapas de väl)
# input resultat av retrieval (queries och dokument), k (samma för alla); output ranking

# en loop med metoderna? kan vara mer manuellt också men det är ju tänkt att de inte ska ta så lång tid
    # börja mäta tid
    # använd rerank-funktionen
    # avsluta tidsmätning
    # börja mäta
    # använd funktion på större/mindre urval (vilken är "bas"?)
    # avsluta mätning
      # byter man till GPU här eller i funktionen? borde typ göras innan man börjar mäta tid
    # börja mäta tid
    # andvänd funktion på GPU
    #sluta mäta
      # tbx till CPU
 # spara alla rankings och tider på ett smidigt sätt
 
 # beräkna NDCG@1, NDCG@10, MRR@5 (t.ex.) med y_true för alla rankings