import React, { useState, useRef, useEffect } from 'react';
import {
  Container,
  TextField,
  Paper,
  Typography,
  Box,
  CircularProgress,
  IconButton,
  Tooltip,
} from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import ArrowForwardIcon from '@mui/icons-material/ArrowForward';
import axios from 'axios';

interface Message {
  text: string;
  isUser: boolean;
  category?: string;
  confidence?: number;
}

function App() {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage: Message = { text: input, isUser: true };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await axios.post('http://localhost:8000/predict', {
        text: input
      });

      const botMessage: Message = {
        text: `Bu haber başlığı "${response.data.category}" kategorisine ait. ( ${(response.data.confidence * 100).toFixed(2)}%)`,
        isUser: false,
        category: response.data.category,
        confidence: response.data.confidence
      };
      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error:', error);
      const errorMessage: Message = {
        text: 'Üzgünüm, bir hata oluştu. Lütfen tekrar deneyin.',
        isUser: false
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const clearChat = () => {
    setMessages([]);
  };

  return (
    <Container maxWidth="md" sx={{ height: '100vh', py: 4 }}>
      <Paper 
        elevation={3} 
        sx={{ 
          height: '100%', 
          display: 'flex', 
          flexDirection: 'column',
          background: 'var(--gradient-surface)',
          borderRadius: 'var(--radius-sm)',
        }}
      >
        <Box 
          sx={{ 
            p: 2, 
            background: 'var(--gradient-primary)',
            color: 'white',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            borderRadius: 'var(--radius-sm) var(--radius-sm) 0 0',
            boxShadow: 'var(--shadow-md)',
          }}
        >
          <Typography variant="h5" component="h1" sx={{ fontWeight: 600 }}>
            Haber Kategorisi Chatbot
          </Typography>
          <Tooltip title="Sohbeti Temizle">
            <IconButton 
              onClick={clearChat} 
              color="inherit"
              sx={{
                '&:hover': {
                  background: 'rgba(255, 255, 255, 0.1)',
                },
              }}
            >
              <DeleteIcon />
            </IconButton>
          </Tooltip>
        </Box>

        <Box 
          sx={{ 
            flex: 1, 
            overflow: 'auto', 
            p: 2,
            background: 'rgba(0, 0, 0, 0.2)',
          }}
        >
          {messages.map((message, index) => (
            <Box
              key={index}
              sx={{
                display: 'flex',
                justifyContent: message.isUser ? 'flex-end' : 'flex-start',
                mb: 2,
              }}
            >
              <Paper
                elevation={1}
                className={message.isUser ? 'message-user' : 'message-bot'}
                sx={{
                  p: 2,
                  maxWidth: '70%',
                  borderRadius: 'var(--radius-md)',
                  backdropFilter: 'blur(10px)',
                }}
              >
                <Typography>{message.text}</Typography>
              </Paper>
            </Box>
          ))}
          {loading && (
            <Box sx={{ display: 'flex', justifyContent: 'center', my: 2 }}>
              <CircularProgress size={24} color="secondary" />
            </Box>
          )}
          <div ref={messagesEndRef} />
        </Box>

        <Box 
          component="form" 
          onSubmit={handleSubmit} 
          sx={{ 
            p: 2, 
            background: 'rgba(0, 0, 0, 0.2)',
            borderRadius: '0 0 var(--radius-sm) var(--radius-sm)',
          }}
        >
          <Box sx={{ display: 'flex', gap: 1 }}>
            <TextField
              fullWidth
              variant="outlined"
              placeholder="Haber başlığını girin..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={loading}
            />
            <Tooltip title="Gönder">
              <IconButton
                type="submit"
                disabled={loading || !input.trim()}
                sx={{ 
                  width: 48,
                  height: 48,
                  background: 'var(--gradient-primary)',
                  '&:hover': {
                    background: 'var(--gradient-hover)',
                    boxShadow: 'var(--shadow-lg)',
                  },
                  '&.Mui-disabled': {
                    background: 'rgba(255, 255, 255, 0.12)',
                  },
                  boxShadow: 'var(--shadow-sm)',
                }}
              >
                <ArrowForwardIcon sx={{ color: 'white' }} />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>
      </Paper>
    </Container>
  );
}

export default App;
