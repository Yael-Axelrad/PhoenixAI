
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

const MAX_HISTORY_MESSAGES = 8

// Allow links only to official Phoenix website pages.
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

function App() {
  // Keep the conversation only while this page is loaded.
  const [isChatOpen, setIsChatOpen] = useState(false)
  const [question, setQuestion] = useState('')
  const [messages, setMessages] = useState<Message[]>([])
  const [loading, setLoading] = useState(false)

  const chatEndRef = useRef<HTMLDivElement>(null)

  // Scroll to the latest message.
  useEffect(() => {
    if (isChatOpen) {
      chatEndRef.current?.scrollIntoView({
        behavior: 'smooth',
        block: 'nearest',
      })
    }
  }, [messages, loading, isChatOpen])

  async function sendMessage(
    event: FormEvent<HTMLFormElement>
  ) {
    event.preventDefault()

    const text = question.trim()

    if (!text || loading) return

    // Capture conversation history before adding the new question.
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

    setMessages([])
    setQuestion('')
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
                onClick={() => setIsChatOpen(false)}
                aria-label="סגירת הצ'אט"
              >
                ×
              </button>
            </div>
          </header>

          <div className="chat-body">
            {messages.length === 0 && (
              <div className="chat-welcome">
                <div className="welcome-symbol">✦</div>

                <h2>שלום! איך אפשר לעזור?</h2>

                <p>
                  אפשר לשאול אותי שאלות כלליות
                  על מוצרי הביטוח והשירותים
                  של הפניקס.
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

            <div ref={chatEndRef} />
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
              placeholder="כתבי שאלה..."
              maxLength={1000}
              aria-label="השאלה שלך"
            />

            <button
              type="submit"
              disabled={!question.trim() || loading}
              aria-label="שליחת שאלה"
            >
              ←
            </button>
          </form>

          <div className="chat-disclaimer">
            מידע כללי בלבד · אינו מהווה ייעוץ ביטוחי
          </div>
        </section>
      )}
    </div>
  )
}

export default App
