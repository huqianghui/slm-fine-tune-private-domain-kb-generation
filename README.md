# slm-fine-tune-private-domain-kb-generation
slm-fine-tune-private-domain-kb-generation


# graph process

1) prompt auto-tuning 

graphrag prompt-tune --config graphProcess/settings.yaml \
--root ./graphragProcess \
--domain "oral cancer" \
--limit 10 \
--chunk-size 2000 \
--discover-entity-types \
--output /home/azureuser/slm-fine-tune-private-domain-kb-generation/graphProcess/output

2) graphrag index

graphrag index --root ./graphragProcess


3) update graphrag with claim output (ValueError: Incremental Indexing Error: No new documents to process.)

graphrag update --config graphProcess/settings.yaml \
--root ./graphragProcess

4) localsearch  query

graphrag query \
--root ./graphragProcess \
--method local \
--query "what stage does T1N1M1 nasopharyngeal carcinoma belong to ?"

5) globalsearch query

graphrag query \
--root ./graphragProcess \
--method global \
--query "what main poits about the staging head and neck cancers?"

6) driftsearch query

graphrag query \
--root ./graphragProcess \
--method drift \
--query "what main poits about the staging head and neck cancers?"