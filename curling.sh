curl --location --request GET 'https://rami-language-testing.cognitiveservices.azure.com/language/analyze-text/jobs?api-version=2022-10-01-preview' \
--header 'Content-Type: application/json' \
--header 'Ocp-Apim-Subscription-Key: 7883b3855a494e77a2780e3ec82cf347' \
--data '{
  "displayName": "Classifying documents",
  "analysisInput": {
    "documents": [
      {
        "id": "1",
        "language": "nb",
        "text": "Text1"
      },
      {
        "id": "2",
        "language": "nb",
        "text": "Text2"
      }
    ]
  },
  "tasks": [
    {
      "kind": "CustomSingleLabelClassification",
      "taskName": "Single Classification Label",
      "parameters": {
        "projectName": "NorwegianLiteratureExample",
        "deploymentName": "NorLitDeployment-1"
      }
    }
  ]
}'