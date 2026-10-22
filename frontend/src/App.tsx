
import {
  useEffect,
  useRef,
  useState,
  type FormEvent,
} from 'react'

import ReactMarkdown from 'react-markdown'
import './App.css'

type Message = {
  role: 'user' | 'assistant'
  text: string
  sources?: string[]
}

type ChatResponse = {
  answer: string
  sources: string[]
}

type SpeechResult = {
  isFinal: boolean
  0: { transcript: string }
}

type SpeechEvent = {
  resultIndex: number
  results: ArrayLike<SpeechResult>
}

type SpeechErrorEvent = {
  error: string
}

type SpeechRecognitionInstance = {
  lang: string
  continuous: boolean
  interimResults: boolean
  onresult: ((event: SpeechEvent) => void) | null
  onerror: ((event: SpeechErrorEvent) => void) | null
  onend: (() => void) | null
  start: () => void
  stop: () => void
}

type SpeechRecognitionConstructor =
  new () => SpeechRecognitionInstance

const MAX_HISTORY_MESSAGES = 8

function isOfficialUrl(url: string): boolean {
  try {
    const parsed = new URL(url)

    return (
      parsed.protocol === 'https:' &&
      parsed.hostname === 'www.fnx.co.il'
    )
  } catch {
    return false
  }
}

function getSpeechRecognition():
  SpeechRecognitionConstructor | undefined {
  const speechWindow = window as Window & {
    SpeechRecognition?: SpeechRecognitionConstructor
    webkitSpeechRecognition?: SpeechRecognitionConstructor
  }

  return (
    speechWindow.SpeechRecognition ??
    speechWindow.webkitSpeechRecognition
  )
}

function App() {
  const [isChatOpen, setIsChatOpen] = useState(false)
  const [question, setQuestion] = useState('')
  const [messages, setMessages] = useState<Message[]>([])
  const [loading, setLoading] = useState(false)

  // Voice recognition state
  const [isListening, setIsListening] = useState(false)
  const [speechError, setSpeechError] = useState('')
  const [speechSupported, setSpeechSupported] = useState(false)
  const [speakingIndex, setSpeakingIndex] = useState<number | null>(null)
  const [ttsSupported, setTtsSupported] = useState(false)

  const chatBodyRef = useRef<HTMLDivElement>(null)
  const recognitionRef =
    useRef<SpeechRecognitionInstance | null>(null)
  const speechBaseTextRef = useRef('')
  const speechFinalTextRef = useRef('')

  useEffect(() => {
    setSpeechSupported(Boolean(getSpeechRecognition()))
    setTtsSupported('speechSynthesis' in window)
  }, [])

  useEffect(() => {
    if (isChatOpen) {
      const body = chatBodyRef.current
      if (body) body.scrollTop = body.scrollHeight
    }
  }, [messages, loading, isChatOpen])

  // Stop recording when the component is removed.
  useEffect(() => {
    return () => {
      window.speechSynthesis?.cancel()
      if (recognitionRef.current) {
        recognitionRef.current.onend = null
        recognitionRef.current.onresult = null
        recognitionRef.current.onerror = null
        recognitionRef.current.stop()
      }
    }
  }, [])

  function stopListening() {
    if (recognitionRef.current) {
      recognitionRef.current.stop()
    }
    setIsListening(false)
  }

  function toggleMicrophone() {
    if (isListening) {
      stopListening()
      return
    }

    if (loading) return

    setSpeechError('')

    const SpeechRecognition = getSpeechRecognition()

    if (!SpeechRecognition) {
      setSpeechError(
        'הדפדפן שלך אינו תומך בזיהוי דיבור. נסי להשתמש ב-Chrome.'
      )
      return
    }

    // Stop any previous recognition session.
    if (recognitionRef.current) {
      recognitionRef.current.onend = null
      recognitionRef.current.stop()
      recognitionRef.current = null
    }

    const recognition = new SpeechRecognition()

    recognition.lang = 'he-IL'
    recognition.continuous = false
    recognition.interimResults = true

    speechBaseTextRef.current = question.trim()
    speechFinalTextRef.current = ''

    recognition.onresult = (event) => {
      let interimText = ''
      let finalText = speechFinalTextRef.current

      for (
        let i = event.resultIndex;
        i < event.results.length;
        i++
      ) {
        const result = event.results[i]
        const transcript = result[0].transcript.trim()

        if (result.isFinal) {
          finalText = [finalText, transcript]
            .filter(Boolean)
            .join(' ')
        } else {
          interimText = [interimText, transcript]
            .filter(Boolean)
            .join(' ')
        }
      }

      speechFinalTextRef.current = finalText

      const combinedText = [
        speechBaseTextRef.current,
        finalText,
        interimText,
      ]
        .filter(Boolean)
        .join(' ')

      setQuestion(combinedText.slice(0, 1000))
    }

    recognition.onerror = (event) => {
      setIsListening(false)

      switch (event.error) {
        case 'not-allowed':
        case 'service-not-allowed':
          setSpeechError(
            'אין הרשאה למיקרופון. יש לאפשר גישה למיקרופון בדפדפן.'
          )
          break

        case 'no-speech':
          setSpeechError(
            'לא זוהה דיבור. נסי לדבר שוב.'
          )
          break

        case 'audio-capture':
          setSpeechError(
            'לא נמצא מיקרופון זמין במכשיר.'
          )
          break

        case 'network':
          setSpeechError(
            'שירות זיהוי הדיבור אינו זמין כרגע. בדקי את החיבור לאינטרנט.'
          )
          break

        case 'aborted':
          break

        default:
          setSpeechError(
            'אירעה תקלה בזיהוי הדיבור. נסי שוב.'
          )
      }
    }

    recognition.onend = () => {
      setIsListening(false)
      recognitionRef.current = null
    }

    recognitionRef.current = recognition

    try {
      recognition.start()
      setIsListening(true)
    } catch {
      setIsListening(false)
      recognitionRef.current = null
      setSpeechError(
        'לא ניתן להפעיל את המיקרופון כרגע.'
      )
    }
  }

  function stopSpeaking() {
    if ('speechSynthesis' in window) window.speechSynthesis.cancel()
    setSpeakingIndex(null)
  }

  function toggleSpeaking(text: string, index: number) {
    if (!('speechSynthesis' in window)) return
    if (speakingIndex === index) {
      stopSpeaking()
      return
    }
    stopSpeaking()
    const cleanText = text
      .replace(/\[Source\s+\d+\]/gi, '')
      .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
      .replace(/https?:\/\/\S+/g, '')
      .replace(/[#*_`>]/g, '')
      .trim()
    if (!cleanText) return
    const utterance = new SpeechSynthesisUtterance(cleanText)
    utterance.lang = 'he-IL'
    utterance.rate = 0.95
    const voices = window.speechSynthesis.getVoices()
    const hebrewVoices = voices.filter((voice) =>
      voice.lang.toLowerCase().startsWith('he')
    )
    // Prefer Hebrew voices with known feminine names when installed.
    const feminineVoice = hebrewVoices.find((voice) =>
      /hila|הילה|carmit|כרמית|yael|יעל|female|woman/i.test(
        `${voice.name} ${voice.voiceURI}`
      )
    )
    const selectedVoice = feminineVoice ?? hebrewVoices[0]
    if (selectedVoice) utterance.voice = selectedVoice
    utterance.onend = () => setSpeakingIndex(null)
    utterance.onerror = () => setSpeakingIndex(null)
    setSpeakingIndex(index)
    window.speechSynthesis.speak(utterance)
  }

  async function sendMessage(
    event: FormEvent<HTMLFormElement>
  ) {
    event.preventDefault()

    if (isListening) {
      stopListening()
    }

    const text = question.trim()

    if (!text || loading) return

    const history = messages
      .slice(-MAX_HISTORY_MESSAGES)
      .map((message) => ({
        role: message.role,
        text: message.text,
      }))

    setMessages((previous) => [
      ...previous,
      {
        role: 'user',
        text,
      },
    ])

    setQuestion('')
    setSpeechError('')
    setLoading(true)

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: text,
          history,
        }),
      })

      if (!response.ok) {
        throw new Error(`API returned ${response.status}`)
      }

      const data: ChatResponse = await response.json()

      setMessages((previous) => [
        ...previous,
        {
          role: 'assistant',
          text: data.answer,
          sources: data.sources,
        },
      ])
    } catch {
      setMessages((previous) => [
        ...previous,
        {
          role: 'assistant',
          text: 'לא הצלחתי לקבל תשובה כרגע. נסי שוב בעוד רגע.',
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  function clearConversation() {
    if (loading) return

    if (isListening) {
      stopListening()
    }

    stopSpeaking()
    setMessages([])
    setQuestion('')
    setSpeechError('')
  }

  function closeChat() {
    stopSpeaking()
    if (isListening) {
      stopListening()
    }

    setIsChatOpen(false)
  }

  return (
    <div className="site" dir="rtl">
      <div className="top-bar">
        <span>שירותים דיגיטליים</span>
        <span>אזור אישי</span>
      </div>

      <header className="site-header">
        <div className="site-brand">
          <img
            src="/PhoenixLogo.png"
            alt="הפניקס"
            className="phoenix-site-logo"
          />
        </div>

        <nav
          className="site-nav"
          aria-label="ניווט ראשי"
        >
          <span>ביטוחים</span>
          <span>פנסיה, גמל וחיסכון</span>
          <span>השקעות</span>
          <span>אשראי</span>
          <span>תביעות</span>
          <span>שירות ופעולות</span>
        </nav>
      </header>

      <main className="site-content">
        <div className="demo-label">
          הדגמת האקתון · אתר לדוגמה
        </div>

        <h1>מידע ביטוחי, פשוט ונגיש יותר</h1>

        <p>
          הכירו את PhoenixAI — עוזר מידע חכם
          שמאפשר לשאול שאלות כלליות על ביטוח
          ולקבל תשובות עם קישורים למקורות
          באתר הפניקס.
        </p>

        <div className="feature-grid">
          <div className="feature-card">
            <span className="feature-icon">✦</span>
            <h2>שאלות בשפה חופשית</h2>
            <p>
              פשוט שואלים, בלי לחפש בין עשרות עמודים.
            </p>
          </div>

          <div className="feature-card">
            <span className="feature-icon">↗</span>
            <h2>תשובות עם מקורות</h2>
            <p>
              קישורים למידע שעליו מבוססת התשובה.
            </p>
          </div>

          <div className="feature-card">
            <span className="feature-icon">✓</span>
            <h2>מידע כללי ללא הזדהות</h2>
            <p>
              אין צורך להזין פרטים אישיים
              כדי לשאול שאלה כללית.
            </p>
          </div>
        </div>
      </main>

      {!isChatOpen && (
        <button
          type="button"
          className="chat-launcher"
          onClick={() => setIsChatOpen(true)}
          aria-label="פתיחת PhoenixAI"
        >
          <span className="launcher-icon">✦</span>
          <span>שאלו את PhoenixAI</span>
        </button>
      )}

      {isChatOpen && (
        <section
          className="chat-window"
          aria-label="צ'אט PhoenixAI"
        >
          <header className="chat-header">
            <div className="chat-heading">
              <span className="chat-avatar">✦</span>

              <div>
                <strong>PhoenixAI</strong>
                <small>עוזר מידע חכם</small>
              </div>
            </div>

            <div className="chat-header-actions">
              <button
                type="button"
                className="clear-chat"
                onClick={clearConversation}
                title="מחיקת היסטוריית השיחה"
                aria-label="מחיקת היסטוריית השיחה"
                disabled={loading}
              >
                ↺
              </button>

              <button
                type="button"
                className="close-chat"
                onClick={closeChat}
                aria-label="סגירת הצ'אט"
              >
                ×
              </button>
            </div>
          </header>

          <div className="chat-body" ref={chatBodyRef}>
            {messages.length === 0 && (
              <div className="chat-welcome">
                <div className="welcome-symbol">✦</div>

                <h2>שלום! איך אפשר לעזור?</h2>

                <p>
                  אפשר לשאול אותי שאלות כלליות
                  על מוצרי הביטוח והשירותים
                  של הפניקס.
                </p>

                <p>
                  אפשר גם ללחוץ על המיקרופון
                  ולשאול שאלה בקול.
                </p>
              </div>
            )}

            {messages.map((message, index) => (
              <div
                key={index}
                className={`chat-message ${message.role}`}
              >
                <div className="message-author">
                  {message.role === 'user'
                    ? 'את'
                    : 'PhoenixAI'}
                </div>

                {message.role === 'assistant' && ttsSupported && (
                  <button
                    type="button"
                    className={`speak-button ${speakingIndex === index ? 'speaking' : ''}`}
                    onClick={() => toggleSpeaking(message.text, index)}
                    aria-label={speakingIndex === index ? 'עצירת הקראה' : 'הקראת התשובה'}
                    title={speakingIndex === index ? 'עצירת הקראה' : 'הקראת התשובה בעברית'}
                  >
{speakingIndex === index ? (
                      <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                        <rect x="6" y="6" width="12" height="12" rx="2" />
                      </svg>
                    ) : (
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                        <path d="M11 5 6 9H3v6h3l5 4V5Z" />
                        <path d="M15.5 8.5a5 5 0 0 1 0 7" />
                        <path d="M18.5 5.5a9 9 0 0 1 0 13" />
                      </svg>
                    )}
                  </button>
                )}

                {message.role === 'assistant' ? (
                  <div className="message-content markdown-content">
                    <ReactMarkdown
                      components={{
                        a: ({ href, children }) =>
                          href && isOfficialUrl(href) ? (
                            <a
                              href={href}
                              target="_blank"
                              rel="noopener noreferrer"
                            >
                              {children}
                            </a>
                          ) : (
                            <span>{children}</span>
                          ),
                      }}
                    >
                      {message.text.replace(
                        /\[Source\s+\d+\]/gi,
                        ''
                      )}
                    </ReactMarkdown>
                  </div>
                ) : (
                  <div className="message-content">
                    {message.text}
                  </div>
                )}

                {message.sources &&
                  message.sources.length > 0 && (
                    <div className="message-sources">
                      <strong>
                        מקורות מאתר הפניקס
                      </strong>

                      {Array.from(
                        new Set(message.sources)
                      )
                        .filter(isOfficialUrl)
                        .map((url, sourceIndex) => (
                          <a
                            key={url}
                            href={url}
                            target="_blank"
                            rel="noopener noreferrer"
                          >
                            ↗ מקור {sourceIndex + 1}
                            {' — '}
                            צפייה בעמוד
                          </a>
                        ))}
                    </div>
                  )}
              </div>
            ))}

            {loading && (
              <div
                className="thinking-indicator"
                role="status"
                aria-label="PhoenixAI חושב"
              >
                <span className="thinking-dot" />
                <span className="thinking-dot" />
                <span className="thinking-dot" />
              </div>
            )}


          </div>

          <form
            className="chat-form"
            onSubmit={sendMessage}
          >
            <input
              value={question}
              onChange={(event) =>
                setQuestion(event.target.value)
              }
              placeholder={
                isListening
                  ? 'מקשיב לך...'
                  : 'כתבי שאלה או לחצי על המיקרופון...'
              }
              maxLength={1000}
              aria-label="השאלה שלך"
            />

            <button
              type="button"
              className={`mic-button ${
                isListening ? 'mic-listening' : ''
              }`}
              onClick={toggleMicrophone}
              disabled={loading || !speechSupported}
              aria-label={
                isListening
                  ? 'עצירת הקלטה'
                  : 'התחלת הקלטה קולית'
              }
              aria-pressed={isListening}
              title={
                !speechSupported
                  ? 'זיהוי דיבור אינו נתמך בדפדפן זה'
                  : isListening
                    ? 'לחצי לעצירת ההאזנה'
                    : 'לחצי כדי לדבר בעברית'
              }
            >
              {isListening ? '■' : '🎤'}
            </button>

            <button
              type="submit"
              disabled={
                !question.trim() ||
                loading ||
                isListening
              }
              aria-label="שליחת שאלה"
            >
              ←
            </button>
          </form>

          {isListening && (
            <div className="voice-status" role="status">
              <span className="voice-pulse" />
              מקשיב לך... לחצי על ■ לסיום
            </div>
          )}

          {speechError && (
            <div className="voice-error" role="alert">
              {speechError}
            </div>
          )}

          <div className="chat-disclaimer">
            מידע כללי בלבד · אינו מהווה ייעוץ ביטוחי
          </div>
        </section>
      )}
    </div>
  )
}

export default App
