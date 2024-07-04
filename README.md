# Backend Augierai.com

This is the backend project for Augierai.com. It provides various functionalities and integrations, including fetching opportunities from SAM.gov and generating proposal drafts using OpenAI's GPT-4 model.

## Features

- Fetch opportunities from SAM.gov
- Generate proposal drafts using GPT-4
- User authentication and management
- API endpoints for interacting with the frontend

## Prerequisites

- Python 3.9 or higher
- pip (Python package installer)
- Node.js and npm (for frontend development)

## Installation

1. **Clone the repository**

   ```sh
   git clone https://github.com/username/backend_augierai.com.git
   cd backend_augierai.com
   pip install -r requirements.txt 

   OPENAI_API_KEY=your_openai_api_key
   SAM_API_KEY=your_sam_api_key
   TAVILY_API_KEY=your_tavily_api_key

  uvicorn main:app --reload

## Run the Front End application augierai.com
cd ../frontend_directory
npm install
npm run dev


## API Endpoints
GET /sam_opportunities: Fetch opportunities from SAM.gov
POST /run_graph: Run the graph to generate proposal drafts


##
POST http://127.0.0.1:8001/run_graph
{
  "task": "Write a bid for a municipal project to upgrade the city's water treatment facilities.",
  "max_revisions": 3
}
