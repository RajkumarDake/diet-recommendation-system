import React, { useState } from 'react';
import axios from 'axios';

const ComprehensiveForm = () => {
  const [formData, setFormData] = useState({
    // Personal Information
    userId: '',
    name: '',
    email: '',
    
    // Health Metrics
    age: '',
    gender: '',
    height: '',
    weight: '',
    cholesterolTotal: '',
    cholesterolHDL: '',
    cholesterolLDL: '',
    bloodPressureSystolic: '',
    bloodPressureDiastolic: '',
    glucose: '',
    triglycerides: '',
    
    // Mental Health
    stressLevel: 5,
    anxietyScore: 5,
    depressionScore: 5,
    sleepQuality: 5,
    moodRating: 5,
    cognitiveFunction: 5,
    socialSupport: 5,
    
    // Lifestyle
    activityLevel: 'moderate',
    exerciseFrequency: 3,
    smokingStatus: 'never',
    alcoholConsumption: 'light',
    dietaryRestrictions: [],
    foodAllergies: [],
    
    // Goals
    goals: ['general_health'],
    
    // Genomic Data (optional)
    genomicFile: null,
    hasGenomicData: false
  });

  const [loading, setLoading] = useState(false);
  const [recommendations, setRecommendations] = useState(null);
  const [error, setError] = useState('');

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    
    if (type === 'checkbox') {
      if (name === 'dietaryRestrictions' || name === 'foodAllergies' || name === 'goals') {
        setFormData(prev => ({
          ...prev,
          [name]: checked 
            ? [...prev[name], value]
            : prev[name].filter(item => item !== value)
        }));
      } else {
        setFormData(prev => ({ ...prev, [name]: checked }));
      }
    } else {
      setFormData(prev => ({ ...prev, [name]: value }));
    }
  };

  const handleFileChange = (e) => {
    setFormData(prev => ({ ...prev, genomicFile: e.target.files[0] }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      // Prepare the request data
      const requestData = {
        user_id: formData.userId || `user_${Date.now()}`,
        health_metrics: {
          age: parseInt(formData.age),
          gender: formData.gender,
          height: parseFloat(formData.height),
          weight: parseFloat(formData.weight),
          cholesterol_total: formData.cholesterolTotal ? parseFloat(formData.cholesterolTotal) : null,
          cholesterol_hdl: formData.cholesterolHDL ? parseFloat(formData.cholesterolHDL) : null,
          cholesterol_ldl: formData.cholesterolLDL ? parseFloat(formData.cholesterolLDL) : null,
          blood_pressure_systolic: formData.bloodPressureSystolic ? parseInt(formData.bloodPressureSystolic) : null,
          blood_pressure_diastolic: formData.bloodPressureDiastolic ? parseInt(formData.bloodPressureDiastolic) : null,
          glucose: formData.glucose ? parseFloat(formData.glucose) : null,
          triglycerides: formData.triglycerides ? parseFloat(formData.triglycerides) : null
        },
        mental_health: {
          stress_level: parseInt(formData.stressLevel),
          anxiety_score: parseInt(formData.anxietyScore),
          depression_score: parseInt(formData.depressionScore),
          sleep_quality: parseInt(formData.sleepQuality),
          mood_rating: parseInt(formData.moodRating),
          cognitive_function: parseInt(formData.cognitiveFunction),
          social_support: parseInt(formData.socialSupport)
        },
        lifestyle: {
          activity_level: formData.activityLevel,
          exercise_frequency: parseInt(formData.exerciseFrequency),
          smoking_status: formData.smokingStatus,
          alcohol_consumption: formData.alcoholConsumption,
          dietary_restrictions: formData.dietaryRestrictions,
          food_allergies: formData.foodAllergies
        },
        goals: formData.goals
      };

      // Add genomic data if available
      if (formData.hasGenomicData && formData.genomicFile) {
        // In a real implementation, you would upload the genomic file first
        // For now, we'll add placeholder genomic data
        requestData.genomic_data = {
          snp_data: {
            "rs1801282": "0/1",
            "rs1801133": "1/1",
            "rs662799": "0/0",
            "rs5918": "0/1",
            "rs1042713": "1/1"
          }
        };
      }

      // Make API request to backend
      const response = await axios.post('http://localhost:8000/api/v1/recommendations/generate', requestData);
      
      setRecommendations(response.data);
      
    } catch (err) {
      console.error('Error generating recommendations:', err);
      setError(err.response?.data?.detail || 'An error occurred while generating recommendations');
    } finally {
      setLoading(false);
    }
  };

  if (recommendations) {
    return <RecommendationResults recommendations={recommendations} onBack={() => setRecommendations(null)} />;
  }

  return (
    <div className="max-w-5xl mx-auto p-8 bg-gradient-to-br from-white to-gray-50 rounded-2xl shadow-2xl border border-gray-100">
      <div className="text-center mb-10">
        <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full mb-4">
          <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
          </svg>
        </div>
        <h2 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-3">
          AI-Powered Personalized Nutrition Assessment
        </h2>
        <p className="text-gray-600 text-lg max-w-2xl mx-auto">
          Comprehensive health analysis using advanced AI models for personalized nutrition recommendations
        </p>
      </div>
      
      <form onSubmit={handleSubmit} className="space-y-8">
        {/* Personal Information */}
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-6 rounded-xl border border-blue-100 shadow-sm">
          <div className="flex items-center mb-4">
            <div className="w-8 h-8 bg-blue-500 rounded-lg flex items-center justify-center mr-3">
              <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
              </svg>
            </div>
            <h3 className="text-xl font-semibold text-gray-800">Personal Information</h3>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <input
              type="text"
              name="name"
              placeholder="Full Name"
              value={formData.name}
              onChange={handleInputChange}
              className="p-4 border-2 border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200 bg-white shadow-sm hover:shadow-md"
              required
            />
            <input
              type="email"
              name="email"
              placeholder="Email Address"
              value={formData.email}
              onChange={handleInputChange}
              className="p-4 border-2 border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200 bg-white shadow-sm hover:shadow-md"
              required
            />
          </div>
        </div>

        {/* Health Metrics */}
        <div className="bg-gradient-to-r from-green-50 to-emerald-50 p-6 rounded-xl border border-green-100 shadow-sm">
          <div className="flex items-center mb-4">
            <div className="w-8 h-8 bg-green-500 rounded-lg flex items-center justify-center mr-3">
              <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
              </svg>
            </div>
            <h3 className="text-xl font-semibold text-gray-800">Health Metrics</h3>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <input
              type="number"
              name="age"
              placeholder="Age"
              value={formData.age}
              onChange={handleInputChange}
              className="p-4 border-2 border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200 bg-white shadow-sm hover:shadow-md"
              required
              min="18"
              max="120"
            />
            <select
              name="gender"
              value={formData.gender}
              onChange={handleInputChange}
              className="p-4 border-2 border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200 bg-white shadow-sm hover:shadow-md"
              required
            >
              <option value="">Select Gender</option>
              <option value="male">Male</option>
              <option value="female">Female</option>
              <option value="other">Other</option>
            </select>
            <input
              type="number"
              name="height"
              placeholder="Height (cm)"
              value={formData.height}
              onChange={handleInputChange}
              className="p-4 border-2 border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200 bg-white shadow-sm hover:shadow-md"
              required
              min="100"
              max="250"
            />
            <input
              type="number"
              name="weight"
              placeholder="Weight (kg)"
              value={formData.weight}
              onChange={handleInputChange}
              className="p-4 border-2 border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200 bg-white shadow-sm hover:shadow-md"
              required
              min="30"
              max="300"
            />
            <input
              type="number"
              name="cholesterolTotal"
              placeholder="Total Cholesterol (mg/dL)"
              value={formData.cholesterolTotal}
              onChange={handleInputChange}
              className="p-4 border-2 border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200 bg-white shadow-sm hover:shadow-md"
            />
            <input
              type="number"
              name="bloodPressureSystolic"
              placeholder="Blood Pressure Systolic"
              value={formData.bloodPressureSystolic}
              onChange={handleInputChange}
              className="p-4 border-2 border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200 bg-white shadow-sm hover:shadow-md"
            />
          </div>
        </div>

        {/* Mental Health Assessment */}
        <div className="bg-gray-50 p-6 rounded-lg">
          <h3 className="text-xl font-semibold mb-4 text-gray-700">Mental Health & Wellness</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {[
              { name: 'stressLevel', label: 'Stress Level' },
              { name: 'anxietyScore', label: 'Anxiety Level' },
              { name: 'depressionScore', label: 'Depression Score' },
              { name: 'sleepQuality', label: 'Sleep Quality' },
              { name: 'moodRating', label: 'Overall Mood' },
              { name: 'cognitiveFunction', label: 'Cognitive Function' },
              { name: 'socialSupport', label: 'Social Support' }
            ].map(({ name, label }) => (
              <div key={name}>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  {label} (1-10): {formData[name]}
                </label>
                <input
                  type="range"
                  name={name}
                  min="1"
                  max="10"
                  value={formData[name]}
                  onChange={handleInputChange}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
              </div>
            ))}
          </div>
        </div>

        {/* Lifestyle Factors */}
        <div className="bg-gray-50 p-6 rounded-lg">
          <h3 className="text-xl font-semibold mb-4 text-gray-700">Lifestyle Factors</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <select
              name="activityLevel"
              value={formData.activityLevel}
              onChange={handleInputChange}
              className="p-4 border-2 border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200 bg-white shadow-sm hover:shadow-md"
            >
              <option value="sedentary">Sedentary</option>
              <option value="light">Light Activity</option>
              <option value="moderate">Moderate Activity</option>
              <option value="active">Active</option>
              <option value="very_active">Very Active</option>
            </select>
            <input
              type="number"
              name="exerciseFrequency"
              placeholder="Exercise Days per Week"
              value={formData.exerciseFrequency}
              onChange={handleInputChange}
              className="p-4 border-2 border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200 bg-white shadow-sm hover:shadow-md"
              min="0"
              max="7"
            />
          </div>
          
          {/* Dietary Restrictions */}
          <div className="mt-4">
            <label className="block text-sm font-medium text-gray-700 mb-2">Dietary Restrictions</label>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
              {['vegetarian', 'vegan', 'gluten_free', 'dairy_free', 'keto', 'paleo'].map(restriction => (
                <label key={restriction} className="flex items-center">
                  <input
                    type="checkbox"
                    name="dietaryRestrictions"
                    value={restriction}
                    checked={formData.dietaryRestrictions.includes(restriction)}
                    onChange={handleInputChange}
                    className="mr-2"
                  />
                  <span className="text-sm capitalize">{restriction.replace('_', ' ')}</span>
                </label>
              ))}
            </div>
          </div>
        </div>

        {/* Genomic Data */}
        <div className="bg-gray-50 p-6 rounded-lg">
          <h3 className="text-xl font-semibold mb-4 text-gray-700">Genomic Data (Optional)</h3>
          <div className="space-y-4">
            <label className="flex items-center">
              <input
                type="checkbox"
                name="hasGenomicData"
                checked={formData.hasGenomicData}
                onChange={handleInputChange}
                className="mr-2"
              />
              <span>I have genomic data (23andMe, AncestryDNA, etc.)</span>
            </label>
            
            {formData.hasGenomicData && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Upload Genomic Data File (VCF format)
                </label>
                <input
                  type="file"
                  accept=".vcf,.txt"
                  onChange={handleFileChange}
                  className="p-3 border rounded-lg focus:ring-2 focus:ring-green-500 w-full"
                />
                <p className="text-sm text-gray-500 mt-1">
                  Supported formats: VCF files from 23andMe, AncestryDNA, or other genetic testing services
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Goals */}
        <div className="bg-gray-50 p-6 rounded-lg">
          <h3 className="text-xl font-semibold mb-4 text-gray-700">Health Goals</h3>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
            {['general_health', 'weight_loss', 'weight_gain', 'muscle_building', 'heart_health', 'diabetes_management'].map(goal => (
              <label key={goal} className="flex items-center">
                <input
                  type="checkbox"
                  name="goals"
                  value={goal}
                  checked={formData.goals.includes(goal)}
                  onChange={handleInputChange}
                  className="mr-2"
                />
                <span className="text-sm capitalize">{goal.replace('_', ' ')}</span>
              </label>
            ))}
          </div>
        </div>

        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
            {error}
          </div>
        )}

        <div className="flex gap-4">
          <button
            type="button"
            onClick={() => window.history.back()}
            className="flex-1 bg-gray-500 text-white py-4 px-6 rounded-xl font-semibold text-lg hover:bg-gray-600 transition-all duration-200 shadow-lg hover:shadow-xl"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={loading}
            className="flex-2 bg-gradient-to-r from-blue-600 to-purple-600 text-white py-4 px-8 rounded-xl font-semibold text-lg hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 shadow-lg hover:shadow-xl transform hover:scale-105 disabled:transform-none"
          >
            {loading ? (
              <div className="flex items-center justify-center">
                <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Generating AI Recommendations...
              </div>
            ) : (
              'Generate Personalized Nutrition Plan'
            )}
          </button>
        </div>
      </form>
    </div>
  );
};

const RecommendationResults = ({ recommendations, onBack }) => {
  return (
    <div className="max-w-6xl mx-auto p-6 bg-white rounded-lg shadow-lg">
      <div className="flex justify-between items-center mb-8">
        <h2 className="text-3xl font-bold text-gray-800">Your Personalized Nutrition Plan</h2>
        <button
          onClick={onBack}
          className="bg-gray-500 text-white px-4 py-2 rounded-lg hover:bg-gray-600"
        >
          Back to Form
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Primary Nutrition Plan */}
        <div className="bg-gradient-to-r from-green-50 to-blue-50 p-6 rounded-lg">
          <h3 className="text-xl font-semibold mb-4 text-gray-800">Primary Nutrition Plan</h3>
          <div className="space-y-3">
            <p><strong>Category:</strong> {recommendations.primary_nutrition_plan?.category}</p>
            <p><strong>Confidence:</strong> {(recommendations.primary_nutrition_plan?.confidence * 100).toFixed(1)}%</p>
            <p className="text-gray-700">{recommendations.primary_nutrition_plan?.description}</p>
          </div>
        </div>

        {/* Meal Timing */}
        <div className="bg-gradient-to-r from-yellow-50 to-orange-50 p-6 rounded-lg">
          <h3 className="text-xl font-semibold mb-4 text-gray-800">Meal Timing</h3>
          <div className="space-y-3">
            <p><strong>Focus:</strong> {recommendations.meal_timing?.focus}</p>
            <p><strong>Confidence:</strong> {(recommendations.meal_timing?.confidence * 100).toFixed(1)}%</p>
          </div>
        </div>

        {/* Supplements */}
        <div className="bg-gradient-to-r from-purple-50 to-pink-50 p-6 rounded-lg">
          <h3 className="text-xl font-semibold mb-4 text-gray-800">Supplement Recommendations</h3>
          {recommendations.supplements?.length > 0 ? (
            <div className="space-y-3">
              {recommendations.supplements.map((supplement, index) => (
                <div key={index} className="border-l-4 border-purple-400 pl-4">
                  <p><strong>{supplement.supplement}</strong></p>
                  <p className="text-sm text-gray-600">{supplement.reason}</p>
                  <p className="text-sm">Confidence: {(supplement.confidence * 100).toFixed(1)}%</p>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-gray-600">No specific supplements recommended at this time.</p>
          )}
        </div>

        {/* Hydration */}
        <div className="bg-gradient-to-r from-blue-50 to-cyan-50 p-6 rounded-lg">
          <h3 className="text-xl font-semibold mb-4 text-gray-800">Hydration Guidance</h3>
          <div className="space-y-3">
            <p><strong>Daily Target:</strong> {recommendations.hydration?.daily_liters?.toFixed(1)} liters</p>
            <ul className="list-disc list-inside text-sm text-gray-700">
              {recommendations.hydration?.recommendations?.map((tip, index) => (
                <li key={index}>{tip}</li>
              ))}
            </ul>
          </div>
        </div>

        {/* Food Recommendations */}
        <div className="bg-gradient-to-r from-green-50 to-teal-50 p-6 rounded-lg lg:col-span-2">
          <h3 className="text-xl font-semibold mb-4 text-gray-800">Food Recommendations</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <h4 className="font-medium text-green-700 mb-2">Recommended Foods</h4>
              <ul className="list-disc list-inside text-sm">
                {recommendations.food_recommendations?.recommended_foods?.map((food, index) => (
                  <li key={index}>{food}</li>
                ))}
              </ul>
            </div>
            <div>
              <h4 className="font-medium text-blue-700 mb-2">Meal Ideas</h4>
              <ul className="list-disc list-inside text-sm">
                {recommendations.food_recommendations?.meal_ideas?.map((meal, index) => (
                  <li key={index}>{meal}</li>
                ))}
              </ul>
            </div>
            <div>
              <h4 className="font-medium text-red-700 mb-2">Foods to Limit</h4>
              <ul className="list-disc list-inside text-sm">
                {recommendations.food_recommendations?.foods_to_limit?.map((food, index) => (
                  <li key={index}>{food}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>

        {/* Sensitivity Alerts */}
        {recommendations.sensitivity_alerts?.length > 0 && (
          <div className="bg-gradient-to-r from-red-50 to-orange-50 p-6 rounded-lg lg:col-span-2">
            <h3 className="text-xl font-semibold mb-4 text-gray-800">Food Sensitivity Alerts</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {recommendations.sensitivity_alerts.map((alert, index) => (
                <div key={index} className="border-l-4 border-red-400 pl-4">
                  <p><strong>{alert.allergen}</strong> - {alert.risk_level} risk</p>
                  <p className="text-sm">Confidence: {(alert.confidence * 100).toFixed(1)}%</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Personalization Score */}
        <div className="bg-gradient-to-r from-indigo-50 to-purple-50 p-6 rounded-lg lg:col-span-2">
          <h3 className="text-xl font-semibold mb-4 text-gray-800">AI Analysis Summary</h3>
          <div className="flex items-center justify-between">
            <div>
              <p><strong>Personalization Score:</strong> {(recommendations.personalization_score * 100).toFixed(1)}%</p>
              <p className="text-sm text-gray-600">Based on multi-modal AI analysis of your health, genetic, and lifestyle data</p>
            </div>
            <div className="text-right">
              <p className="text-sm text-gray-600">Generated: {new Date(recommendations.timestamp).toLocaleDateString()}</p>
              <p className="text-sm text-gray-600">Next Update: {new Date(recommendations.next_update_date).toLocaleDateString()}</p>
            </div>
          </div>
        </div>
      </div>

      <div className="mt-8 p-4 bg-gray-100 rounded-lg">
        <p className="text-sm text-gray-600 text-center">
          <strong>Disclaimer:</strong> These recommendations are generated by AI models and should not replace professional medical advice. 
          Please consult with healthcare providers before making significant dietary changes.
        </p>
      </div>
    </div>
  );
};

export default ComprehensiveForm;
