import React, { useState, useEffect, useRef } from 'react';
import { Send, AlertTriangle, Clock, User, Phone, MessageCircle } from 'lucide-react';

function App() {
  const [activeTab, setActiveTab] = useState('chat');
  const [message, setMessage] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [currentUser, setCurrentUser] = useState({
    id: 'user123',
    name: 'Anonymous User',
    priority: 0,
    riskLevel: 'Unknown',
    joinedAt: new Date()
  });
  const [queue, setQueue] = useState([]);
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const chatEndRef = useRef(null);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [chatHistory]);

  const getSeverityColor = (severity) => {
    if (severity.includes('Very Severe (Suicide Risk)')) return 'bg-red-600';
    if (severity.includes('Very Severe')) return 'bg-red-500';
    if (severity.includes('Severe')) return 'bg-orange-500';
    if (severity.includes('Moderate')) return 'bg-yellow-500';
    return 'bg-green-500';
  };

  const getSeverityIcon = (severity) => {
    if (severity.includes('Suicide Risk')) return '🚨';
    if (severity.includes('Very Severe')) return '🔴';
    if (severity.includes('Severe')) return '🟠';
    if (severity.includes('Moderate')) return '🟡';
    return '🟢';
  };

  const handleSendMessage = async () => {
    if (!message.trim()) return;

    setIsLoading(true);
    const userMessage = {
      id: Date.now(),
      text: message,
      sender: 'user',
      timestamp: new Date()
    };

    setChatHistory(prev => [...prev, userMessage]);
    setMessage('');

    try {
      const res = await fetch('http://localhost:8000/api/message', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message: message, user_id: currentUser.id })
      });

      if (!res.ok) throw new Error('Failed to analyze message');
      const data = await res.json();

      // Update current user's risk assessment
      setCurrentUser(prev => ({
        ...prev,
        priority: data.priority_score,
        riskLevel: data.severity,
        lastMessage: message,
        emotions: data.top_emotions
      }));

      // Add AI response to chat
      const aiMessage = {
        id: Date.now() + 1,
        text: data.ai_response,
        sender: 'ai',
        timestamp: new Date(),
        analysis: data
      };

      setChatHistory(prev => [...prev, aiMessage]);

      // Update queue with current user
      setQueue(prev => {
        const existing = prev.find(u => u.id === currentUser.id);
        const updatedUser = {
          ...currentUser,
          priority: data.priority_score,
          riskLevel: data.severity,
          lastMessage: message,
          emotions: data.top_emotions,
          lastActivity: new Date()
        };

        if (existing) {
          return prev.map(u => u.id === currentUser.id ? updatedUser : u)
                    .sort((a, b) => b.priority - a.priority);
        } else {
          return [...prev, updatedUser].sort((a, b) => b.priority - a.priority);
        }
      });

      setError('');
    } catch (err) {
      console.error(err);
      setError('Server error. Check if backend is running.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const formatTime = (date) => {
    return new Intl.DateTimeFormat('en-US', {
      hour: '2-digit',
      minute: '2-digit'
    }).format(date);
  };

  const getWaitTime = (joinedAt) => {
    const diff = new Date() - joinedAt;
    const minutes = Math.floor(diff / 60000);
    return minutes < 1 ? 'Just now' : `${minutes}m`;
  };

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-6xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-blue-500 rounded-full flex items-center justify-center">
                <MessageCircle className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Mental Health Triage</h1>
                <p className="text-sm text-gray-500">AI-Powered Crisis Support System</p>
              </div>
            </div>
            <div className="flex space-x-2">
              <button
                onClick={() => setActiveTab('chat')}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  activeTab === 'chat' 
                    ? 'bg-blue-500 text-white' 
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
              >
                Chat Interface
              </button>
              <button
                onClick={() => setActiveTab('queue')}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  activeTab === 'queue' 
                    ? 'bg-blue-500 text-white' 
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
              >
                Priority Queue ({queue.length})
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-4 py-6">
        {activeTab === 'chat' ? (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Chat Interface */}
            <div className="lg:col-span-2 bg-white rounded-lg shadow-sm border">
              <div className="p-4 border-b bg-gray-50">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center">
                      <User className="w-5 h-5 text-white" />
                    </div>
                    <div>
                      <h3 className="font-medium text-gray-900">{currentUser.name}</h3>
                      <p className="text-sm text-gray-500">ID: {currentUser.id}</p>
                    </div>
                  </div>
                  {currentUser.priority > 0 && (
                    <div className={`px-3 py-1 rounded-full text-white text-sm font-medium ${getSeverityColor(currentUser.riskLevel)}`}>
                      {getSeverityIcon(currentUser.riskLevel)} Priority: {currentUser.priority}
                    </div>
                  )}
                </div>
              </div>

              {/* Chat Messages */}
              <div className="h-96 overflow-y-auto p-4 space-y-4">
                {chatHistory.length === 0 ? (
                  <div className="text-center text-gray-500 py-8">
                    <MessageCircle className="w-12 h-12 mx-auto mb-4 text-gray-300" />
                    <p>Start a conversation to begin crisis assessment</p>
                  </div>
                ) : (
                  chatHistory.map((msg) => (
                    <div
                      key={msg.id}
                      className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}
                    >
                      <div className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                        msg.sender === 'user' 
                          ? 'bg-blue-500 text-white' 
                          : 'bg-gray-200 text-gray-800'
                      }`}>
                        <p className="text-sm">{msg.text}</p>
                        <p className={`text-xs mt-1 ${
                          msg.sender === 'user' ? 'text-blue-100' : 'text-gray-500'
                        }`}>
                          {formatTime(msg.timestamp)}
                        </p>
                      </div>
                    </div>
                  ))
                )}
                <div ref={chatEndRef} />
              </div>

              {/* Message Input */}
              <div className="p-4 border-t">
                {error && (
                  <div className="mb-3 p-3 bg-red-50 border border-red-200 rounded-lg">
                    <p className="text-sm text-red-600">{error}</p>
                  </div>
                )}
                <div className="flex space-x-2">
                  <textarea
                    value={message}
                    onChange={(e) => setMessage(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="Type your message here..."
                    className="flex-1 p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                    rows="2"
                    disabled={isLoading}
                  />
                  <button
                    onClick={handleSendMessage}
                    disabled={isLoading || !message.trim()}
                    className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
                  >
                    <Send className="w-5 h-5" />
                  </button>
                </div>
              </div>
            </div>

            {/* Current User Status */}
            <div className="bg-white rounded-lg shadow-sm border p-4">
              <h3 className="font-medium text-gray-900 mb-4">Current Assessment</h3>
              
              {currentUser.priority > 0 ? (
                <div className="space-y-3">
                  <div className={`p-3 rounded-lg text-white ${getSeverityColor(currentUser.riskLevel)}`}>
                    <div className="flex items-center space-x-2">
                      <span className="text-lg">{getSeverityIcon(currentUser.riskLevel)}</span>
                      <span className="font-medium">{currentUser.riskLevel}</span>
                    </div>
                    <p className="text-sm mt-1">Priority Score: {currentUser.priority}</p>
                  </div>
                  
                  {currentUser.emotions && (
                    <div className="p-3 bg-gray-50 rounded-lg">
                      <p className="text-sm font-medium text-gray-700 mb-2">Top Emotions:</p>
                      <div className="space-y-1">
                        {currentUser.emotions.map((emotion, idx) => (
                          <div key={idx} className="flex justify-between text-sm">
                            <span className="capitalize">{emotion.label}</span>
                            <span className="text-gray-600">{(emotion.score * 100).toFixed(1)}%</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-center text-gray-500 py-4">
                  <AlertTriangle className="w-8 h-8 mx-auto mb-2 text-gray-300" />
                  <p className="text-sm">No assessment yet</p>
                </div>
              )}
            </div>
          </div>
        ) : (
          /* Priority Queue */
          <div className="bg-white rounded-lg shadow-sm border">
            <div className="p-4 border-b">
              <h3 className="font-medium text-gray-900">Priority Queue - Crisis Support</h3>
              <p className="text-sm text-gray-500 mt-1">Users sorted by urgency and risk level</p>
            </div>
            
            <div className="divide-y">
              {queue.length === 0 ? (
                <div className="text-center text-gray-500 py-12">
                  <Clock className="w-12 h-12 mx-auto mb-4 text-gray-300" />
                  <p>No users in queue</p>
                </div>
              ) : (
                queue.map((user, index) => (
                  <div key={user.id} className="p-4 hover:bg-gray-50 transition-colors">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <div className="flex-shrink-0">
                          <div className="w-10 h-10 bg-gray-500 rounded-full flex items-center justify-center">
                            <User className="w-6 h-6 text-white" />
                          </div>
                        </div>
                        <div className="flex-1">
                          <div className="flex items-center space-x-2">
                            <p className="font-medium text-gray-900">{user.name}</p>
                            <span className="text-sm text-gray-500">#{index + 1}</span>
                          </div>
                          <p className="text-sm text-gray-500">ID: {user.id}</p>
                          {user.lastMessage && (
                            <p className="text-sm text-gray-600 mt-1 truncate max-w-md">
                              "{user.lastMessage}"
                            </p>
                          )}
                        </div>
                      </div>
                      
                      <div className="flex items-center space-x-4">
                        <div className="text-right">
                          <div className={`px-3 py-1 rounded-full text-white text-sm font-medium ${getSeverityColor(user.riskLevel)}`}>
                            {getSeverityIcon(user.riskLevel)} {user.priority}
                          </div>
                          <p className="text-xs text-gray-500 mt-1">
                            Wait: {getWaitTime(user.joinedAt)}
                          </p>
                        </div>
                        
                        <button className="px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors flex items-center space-x-1">
                          <Phone className="w-4 h-4" />
                          <span>Connect</span>
                        </button>
                      </div>
                    </div>
                    
                    {user.emotions && (
                      <div className="mt-3 flex space-x-4 text-sm text-gray-600">
                        {user.emotions.map((emotion, idx) => (
                          <span key={idx} className="capitalize">
                            {emotion.label}: {(emotion.score * 100).toFixed(1)}%
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                ))
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;