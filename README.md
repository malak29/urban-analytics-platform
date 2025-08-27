# Urban Analytics Platform

A modern data visualization platform for urban and educational analytics, featuring interactive dashboards for noise complaint analysis and higher education metrics.

## Architecture Overview

```mermaid
graph TB
    subgraph "Data Sources"
        A[NYC Open Data API]
        B[Education Datasets]
        C[CSV/Excel Files]
        D[Real-time Feeds]
    end

    subgraph "Data Processing Layer"
        E[Python ETL Pipeline]
        F[Data Validation]
        G[Feature Engineering]
    end

    subgraph "Storage Layer"
        H[(PostgreSQL Database)]
        I[(Redis Cache)]
        J[File Storage]
    end

    subgraph "API Layer"
        K[FastAPI Backend]
        L[GraphQL Endpoints]
        M[WebSocket Server]
    end

    subgraph "Frontend Applications"
        N[Next.js Dashboard]
        O[React Components]
        P[D3.js Visualizations]
    end

    subgraph "Infrastructure"
        Q[Docker Containers]
        R[Kubernetes Cluster]
        S[CI/CD Pipeline]
    end

    A --> E
    B --> E
    C --> E
    D --> E
    
    E --> F
    F --> G
    G --> H
    G --> I
    G --> J
    
    H --> K
    I --> K
    J --> K
    
    K --> L
    K --> M
    L --> N
    M --> N
    
    N --> O
    O --> P
    
    Q --> R
    R --> S
```

## Core Features

### Urban Analytics Module
- **NYC Noise Analysis**: Interactive maps and temporal analysis of 3M+ noise complaints
- **Geographic Hotspots**: Heat map visualization of complaint density
- **Temporal Patterns**: Time-series analysis with peak hour identification
- **Complaint Categorization**: Breakdown by noise type and severity

### Education Analytics Module
- **Institution Overview**: Comprehensive metrics for higher education institutions
- **Enrollment Analysis**: Student demographics and diversity statistics
- **Financial Metrics**: Cost analysis and financial aid distribution
- **Outcome Tracking**: Graduation rates and post-graduation statistics

## Technology Stack

### Backend
- **API Framework**: FastAPI with Pydantic models
- **Database**: PostgreSQL 15 with TimescaleDB extension
- **Cache**: Redis for session management and query caching
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Task Queue**: Celery with Redis broker

### Frontend
- **Framework**: Next.js 14 with TypeScript
- **UI Components**: Shadcn/UI with Tailwind CSS
- **Visualization**: D3.js, Plotly.js, Recharts
- **State Management**: Zustand
- **Authentication**: NextAuth.js

### Infrastructure
- **Containerization**: Docker with multi-stage builds
- **Orchestration**: Kubernetes with Helm charts
- **CI/CD**: GitHub Actions
- **Monitoring**: Prometheus + Grafana
- **Deployment**: AWS EKS or Google GKE

## Quick Start

### Prerequisites
- Docker 24+
- Node.js 20+
- Python 3.11+

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/urban-analytics-platform
cd urban-analytics-platform

# Start services
docker-compose up -d

# Install frontend dependencies
cd frontend && npm install

# Install backend dependencies
cd ../backend && pip install -r requirements.txt

# Run database migrations
python manage.py migrate

# Start development servers
npm run dev  # Frontend (port 3000)
python main.py  # Backend (port 8000)
```

### Environment Variables

```env
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/analytics
REDIS_URL=redis://localhost:6379

# APIs
NYC_OPEN_DATA_API_KEY=your_api_key
EDUCATION_DATA_API_KEY=your_api_key

# Authentication
NEXTAUTH_SECRET=your_secret_key
NEXTAUTH_URL=http://localhost:3000
```

## Project Results

### NYC Noise Analysis Findings

**Dataset**: 3+ million noise complaints analyzed from 2010-2024

#### Geographic Hotspots
- **Primary hotspots**: Lower Manhattan, Upper East Side, Western Brooklyn
- **Peak complaint areas**: Some zip codes reporting over 3,400 complaints
- **Population correlation**: Strong relationship between population density and noise levels

#### Complaint Patterns
- **Peak complaint hours**: Evening (7 PM - 11 PM)
- **Most persistent issue**: Vehicle noise (consistent throughout day)
- **Complaint distribution**:
  - Illegal Parking: 27.5%
  - Residential Noise: 19.5%
  - Water Issues: 12.8%
  - Blocked Driveways: 9.2%
  - Street/Sidewalk Noise: 9.1%

#### Temporal Analysis
- Vehicle noise remains constant across all hours
- Helicopter noise shows specific time-based patterns
- Clear weekly patterns with consistent complaint volumes Monday-Sunday

### Higher Education Dashboard

**Link**: [Higher Education Dashboard](https://public.tableau.com/views/HigheredDashboard/Dashboard1?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

#### Data Variables Selected
- Institution details (name, location, type)
- Enrollment figures and demographics
- Diversity statistics and representation
- Financial metrics and aid distribution
- Outcome measures and graduation rates

#### Design Implementation
- Interactive filtering by geographic location
- Dynamic visualizations with institutional comparisons
- City-based filtering for regional analysis
- Comprehensive institutional profiling system

**Author**: Malak Parmar | **Institution**: Northeastern University  
**Course**: EAI 6120 AI Visualization and Communication | **Instructor**: Prof. Walter McHugh
