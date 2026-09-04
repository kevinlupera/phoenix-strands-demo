# agent knowledge base
echo "deploying knowledge base ..."
python3 prereqs/knowledge_base.py --mode create

# agent dynamodb
echo "deploying DynamoDB ..."
python3 prereqs/dynamodb.py --mode create


