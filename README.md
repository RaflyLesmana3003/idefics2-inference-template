# idefics2-inference-template
interence tempte for deploying idefics2 model

custom inference endpoint for fine-tuned idefics2 8b

input contain base64 image or file
request example:
```
curl "https://[endpoint].endpoints.huggingface.cloud" \
-X POST \
-H "Accept: application/json" \
-H "Authorization: Bearer [huggingface token]" \
-H "Content-Type: application/json" \
-d '{
    "inputs": "iVBORw0KGgoAAAANSUhEUgAAAnQAAAGdCAYAAACbyu4YAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAJRiSURBVHhe7Z0JuE7V98d3s4qUVKRQCGUsSvJXNFCRNGeIBipDKg0/TSiiQYRSyRBKCJUMRYMGIik"
}'
```
