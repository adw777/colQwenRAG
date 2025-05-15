# import requests
# import json

# url = "https://arxiv.org/search/?searchtype=all&query=AI+in+Legal+Analysis&abstracts=show&size=200&order=-announced_date_first"
# headers = {
#     "Accept": "application/json",
#     "Authorization": "Bearer jina_c4a1ef52b7c44e338de03e985f437581WWcIpdcBib6IlWutDZIsefuHcTIN",
# }

# response = requests.get(url, headers=headers)

# # Check the response status code
# if response.status_code == 200:
#     try:
#         data = response.json()
        
#         # Save as JSON file
#         with open('arxiv_results.json', 'w', encoding='utf-8') as f:
#             json.dump(data, f, indent=4)

#         # Save as Markdown file
#         with open('arxiv_results.md', 'w', encoding='utf-8') as f:
#             f.write(str(data))  # You might want to format this differently depending on your needs

#         print("Results saved to arxiv_results.json and arxiv_results.md")
#     except json.JSONDecodeError:
#         print("Error decoding JSON. Response text:", response.text)
# else:
#     print(f"Error: {response.status_code}")


import requests
import json

url = "https://arxiv.org/search/?searchtype=all&query=AI+in+Legal+Analysis&abstracts=show&size=200&order=-announced_date_first"
headers = {
    "Accept": "application/json",
    "Authorization": "Bearer jina_c4a1ef52b7c44e338de03e985f437581WWcIpdcBib6IlWutDZIsefuHcTIN",
}

response = requests.get(url, headers=headers)

# Check the response status code
if response.status_code == 200:
    # Save the raw response content as an HTML file
    with open('arxiv_results.html', 'w', encoding='utf-8') as f:
        f.write(response.text)

    print("Results saved to arxiv_results.html")
else:
    print(f"Error: {response.status_code}, Response: {response.text}")