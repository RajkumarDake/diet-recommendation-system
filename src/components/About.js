import React from "react";
import { Container, Row, Col, Card } from "react-bootstrap";

function About() {
  return (
    <section id="about" className="py-5 bg-light">
      <Container>
        <Row className="justify-content-center">
          <Col lg={10}>
            <div className="text-center mb-5">
              <h2 className="display-4 fw-bold text-primary mb-3">
                About Our AI-Powered Nutrition System
              </h2>
              <p className="lead text-muted">
                Transforming personalized nutrition through advanced AI and multi-modal data integration
              </p>
            </div>
          </Col>
        </Row>

        <Row className="g-4">
          <Col md={6} lg={4}>
            <Card className="h-100 shadow-sm border-0">
              <Card.Body className="text-center p-4">
                <div className="mb-3">
                  <i className="fas fa-brain fa-3x text-primary"></i>
                </div>
                <Card.Title className="h4 mb-3">Multi-Modal AI</Card.Title>
                <Card.Text className="text-muted">
                  Advanced LSTM, Transformer, and Fusion models analyze your health data, 
                  genomic information, and mental health metrics for comprehensive insights.
                </Card.Text>
              </Card.Body>
            </Card>
          </Col>

          <Col md={6} lg={4}>
            <Card className="h-100 shadow-sm border-0">
              <Card.Body className="text-center p-4">
                <div className="mb-3">
                  <i className="fas fa-dna fa-3x text-success"></i>
                </div>
                <Card.Title className="h4 mb-3">Genomic Integration</Card.Title>
                <Card.Text className="text-muted">
                  Incorporates SNP data from dbSNP and genomic variations to provide 
                  personalized nutrition recommendations based on your genetic profile.
                </Card.Text>
              </Card.Body>
            </Card>
          </Col>

          <Col md={6} lg={4}>
            <Card className="h-100 shadow-sm border-0">
              <Card.Body className="text-center p-4">
                <div className="mb-3">
                  <i className="fas fa-chart-line fa-3x text-info"></i>
                </div>
                <Card.Title className="h4 mb-3">Real-Time Adaptation</Card.Title>
                <Card.Text className="text-muted">
                  Continuously adapts recommendations based on your evolving health metrics, 
                  lifestyle changes, and nutritional responses.
                </Card.Text>
              </Card.Body>
            </Card>
          </Col>

          <Col md={6} lg={4}>
            <Card className="h-100 shadow-sm border-0">
              <Card.Body className="text-center p-4">
                <div className="mb-3">
                  <i className="fas fa-database fa-3x text-warning"></i>
                </div>
                <Card.Title className="h4 mb-3">Comprehensive Data</Card.Title>
                <Card.Text className="text-muted">
                  Integrates data from NHANES, UK Biobank, USDA Nutrient Database, 
                  and DrugBank for evidence-based recommendations.
                </Card.Text>
              </Card.Body>
            </Card>
          </Col>

          <Col md={6} lg={4}>
            <Card className="h-100 shadow-sm border-0">
              <Card.Body className="text-center p-4">
                <div className="mb-3">
                  <i className="fas fa-shield-alt fa-3x text-danger"></i>
                </div>
                <Card.Title className="h4 mb-3">Drug Interactions</Card.Title>
                <Card.Text className="text-muted">
                  Advanced analysis of drug-nutrient interactions to ensure safe and 
                  effective personalized nutrition planning.
                </Card.Text>
              </Card.Body>
            </Card>
          </Col>

          <Col md={6} lg={4}>
            <Card className="h-100 shadow-sm border-0">
              <Card.Body className="text-center p-4">
                <div className="mb-3">
                  <i className="fas fa-microscope fa-3x text-secondary"></i>
                </div>
                <Card.Title className="h4 mb-3">Research Grade</Card.Title>
                <Card.Text className="text-muted">
                  Built as a Master's research project with state-of-the-art AI models 
                  and rigorous scientific methodology.
                </Card.Text>
              </Card.Body>
            </Card>
          </Col>
        </Row>

        <Row className="mt-5">
          <Col lg={8} className="mx-auto">
            <div className="text-center">
              <h3 className="h4 mb-3">How It Works</h3>
              <p className="text-muted mb-4">
                Our system combines cutting-edge AI with comprehensive health data to deliver 
                personalized nutrition recommendations that adapt to your unique biological profile.
              </p>
              <div className="d-flex justify-content-center flex-wrap gap-3">
                <span className="badge bg-primary fs-6 px-3 py-2">LSTM Health Analysis</span>
                <span className="badge bg-success fs-6 px-3 py-2">Genomic Processing</span>
                <span className="badge bg-info fs-6 px-3 py-2">Mental Health Integration</span>
                <span className="badge bg-warning fs-6 px-3 py-2">Fusion Modeling</span>
              </div>
            </div>
          </Col>
        </Row>
      </Container>
    </section>
  );
}

export default About;
