# SWC Poker — Reverse Engineering Research

## Domain Info
- Landing page: https://swcpoker.club (Gatsby static)
- App: https://play.swcpoker.club (React SPA)
- App version: 6.9.18 (hash: 1773176104576)
- Config: https://play.swcpoker.club/config.json

## Server Endpoints
| Key | URL |
|-----|-----|
| nodeProxyUrl | https://game.swcpoker.club:443/poker/ |
| masterProxyUrl | https://game.swcpoker.club/html5poker-services/ |
| loggerServiceUrl | https://game.swcpoker.club/html5poker-services/logger |
| realIpServiceUrl | https://game.swcpoker.club/userservice/realip |

## Login Flow
1. Browser fetches config.json → gets all endpoint URLs
2. POST to nodeProxyUrl + "__JqDt" (login endpoint path) with {login, password}
3. Server returns token
4. WebSocket connect to nodeProxyUrl + "_9N1jp" (game namespace) with token
5. Guest login: nodeProxyUrl + "_9N1jp" via CreateTokenGuest

## WebSocket Protocol
- Transport: Socket.io
- Message format: JSON with unique callback IDs
- Auth: token-based, sent on connect
- Keep-alive: "client-ping" messages

## Key Messages (from JS source)
- CreateToken / CreateTokenGuest → authentication
- PlayerCommand → game actions (fold, call, raise, bet, allin)
- TableAllowedSeats → lobby/table data
- miniConfigXML / convertersXML → received after login (XML-based config)

## Game Features
- Table types: cash games, tournaments, sit-n-go
- Games: Texas Holdem, Omaha, Draw, Stud, Mixed
- Currency: Bitcoin (1 BTC = 1,000,000 chips)
- Anonymous accounts, no KYC

## Anti-Bot Measures
- Fingerprint2.js (browser fingerprinting) — loaded in app.js
- realIpServiceUrl → IP detection/blocking
- Datacenter IPs likely blocked (need residential proxy)
- Token-based session management

## Tech Stack
- Frontend: React + Redux, Socket.io client, Dojo Toolkit, Zend Framework (backend)
- Backend (masterProxyUrl): Zend Framework PHP
- Real-time: Node.js (nodeProxyUrl) with Socket.io
- Storage: localStorage for userData/session