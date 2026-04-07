# SICT Bimini Web

KBIMS 부위코드 예측 웹 애플리케이션

## Quick Start

```bash
# 1. 의존성 설치
bun install

# 2. 환경변수 설정
cp .env.example .env.local
# BACKEND_URL에 FastAPI 서버 주소 설정

# 3. 개발 서버 실행
bun dev
```

> **Note**: 이 애플리케이션은 FastAPI 백엔드 서버가 필요합니다.

## 기술 스택

| 카테고리                | 기술                                          | 버전    |
| ----------------------- | --------------------------------------------- | ------- |
| Runtime/Package Manager | [Bun](https://bun.sh)                         | -       |
| Framework               | [Next.js](https://nextjs.org) (App Router)    | 16.1.1  |
| UI Library              | [React](https://react.dev)                    | 19.2.3  |
| CSS Framework           | [Tailwind CSS](https://tailwindcss.com)       | v4      |
| UI Components           | [Shadcn UI](https://ui.shadcn.com) (new-york) | -       |
| Date Library            | [Day.js](https://day.js.org)                  | 1.11.19 |

## 아키텍처

[Feature-Sliced Design](https://feature-sliced.design/) 패턴 적용:

```
src/
├── 1app/       # 전역 스타일, 프로바이더
├── 2pages/     # 페이지 컴포넌트
├── 3widgets/   # 독립적 UI 블록
├── 4features/  # 사용자 기능
├── 5entities/  # 비즈니스 엔티티
└── 6shared/    # 공통 유틸리티, UI 컴포넌트
```

## 개발

| 명령어                   | 설명                 |
| ------------------------ | -------------------- |
| `bun dev`                | 개발 서버 실행       |
| `bun run build`          | 프로덕션 빌드        |
| `bun lint`               | ESLint 검사          |
| `bunx shadcn@latest add` | Shadcn 컴포넌트 추가 |
